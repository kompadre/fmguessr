package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/kompadre/fmguessr/external/msfa"
	"github.com/kompadre/fmguessr/pkg/match"
)

func main() {
	codebookPath := flag.String("codebook", "dx7_codebook.bin", "Path to codebook binary")
	outDir := flag.String("outdir", "rendered", "Output directory for WAV files")
	noteNum := flag.Int("note", 48, "MIDI note to render (default: 48 = C3)")
	noteMin := flag.Int("note-min", 0, "Minimum MIDI note for multi-note rendering (0=single note mode)")
	noteMax := flag.Int("note-max", 0, "Maximum MIDI note for multi-note rendering (0=single note mode)")
	variants := flag.Int("variants", 50, "Total samples per patch per note (1=base only, no trimming)")
	flag.Parse()

	os.MkdirAll(*outDir, 0755)

	if *variants < 1 {
		fmt.Fprintf(os.Stderr, "Error: --variants must be >= 1\n")
		os.Exit(1)
	}

	// Build note list
	var notes []int
	if *noteMin > 0 && *noteMax > 0 && *noteMax >= *noteMin {
		for n := *noteMin; n <= *noteMax; n++ {
			notes = append(notes, n)
		}
	} else {
		notes = []int{*noteNum}
	}

	// Load codebook (supports both standard DXCB and timbre DXCT formats)
	cb, err := match.LoadCodebookAny(*codebookPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading codebook: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d patches from %s\n", len(cb.Entries), *codebookPath)
	fmt.Printf("Rendering %d samples per patch at MIDI %v (parallel)...\n", *variants, notes)

	// Render duration determines trim range
	renderLen := 2 * match.RenderSampleRate // 2s from renderWithEnvelope
	maxTrimSamples := match.RenderSampleRate / 2 // 500ms max per side
	minRetainedSamples := match.RenderSampleRate  // 1s minimum retained

	// Build work items: base + temporal-trimmed variants
	type workItem struct {
		flatIdx   int
		patchIdx  int
		variant   int
		note      int
		cand      *match.Candidate // always the ORIGINAL candidate
		trimStart int              // samples to trim from start (0 = no trim)
		trimEnd   int              // samples to trim from end (0 = no trim)
	}

	var workItems []workItem
	rng := rand.New(rand.NewSource(42))
	flatIdx := 0

	for patchIdx, entry := range cb.Entries {
		orig := entry.Candidate
		for v := 0; v < *variants; v++ {
			var trimStart, trimEnd int
			if v > 0 {
				// Random temporal trim
				trimStart = rng.Intn(maxTrimSamples + 1)
				// Ensure we keep at least minRetainedSamples
				maxEndTrim := maxTrimSamples
				remaining := renderLen - trimStart - minRetainedSamples
				if remaining < maxEndTrim {
					maxEndTrim = remaining
				}
				if maxEndTrim < 0 {
					maxEndTrim = 0
				}
				trimEnd = rng.Intn(maxEndTrim + 1)
			}
			for _, note := range notes {
				workItems = append(workItems, workItem{
					flatIdx:  flatIdx,
					patchIdx: patchIdx,
					variant:  v,
					note:     note,
					cand:     orig,
					trimStart: trimStart,
					trimEnd:   trimEnd,
				})
				flatIdx++
			}
		}
	}

	totalSamples := len(workItems)
	fmt.Printf("Total work items: %d (%d patches × %d samples × %d notes)\n",
		totalSamples, len(cb.Entries), *variants, len(notes))

	// Parallel rendering with worker pool
	numWorkers := runtime.GOMAXPROCS(0)
	fmt.Printf("Using %d workers\n", numWorkers)

	var mu sync.Mutex
	completed := 0

	// Thread-safe label collection
	allLabels := make([][]float32, totalSamples)

	// Channel for work items
	workChan := make(chan workItem, numWorkers*4)

	// Start workers BEFORE filling the channel to avoid deadlock
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// Each worker gets its own synth instance
			workerSynth := msfa.NewSynth(float64(match.RenderSampleRate))
			defer workerSynth.Free()

			for item := range workChan {
				// Render original patch (labels always match codebook exactly)
				pcm := renderWithEnvelope(workerSynth, item.cand, item.note)

				// Apply temporal trim (simulates target.wav truncation)
				if item.trimStart > 0 || item.trimEnd > 0 {
					end := len(pcm) - item.trimEnd
					if end > item.trimStart {
						pcm = pcm[item.trimStart:end]
					}
				}

				// Save WAV
				wavPath := fmt.Sprintf("%s/p%04d_v%02d_n%03d.wav", *outDir, item.patchIdx, item.variant, item.note)
				saveWav(wavPath, pcm, match.RenderSampleRate)

				// Store label: always the original codebook patch params
				label := normalizeTimbreParams(item.cand)
				allLabels[item.flatIdx] = label

				mu.Lock()
				completed++
				patchProgress := (item.patchIdx + 1)
				if item.variant == 0 && item.note == notes[0] {
					if patchProgress%100 == 0 || patchProgress == len(cb.Entries) {
						fmt.Printf("  [%d/%d] %s\n", patchProgress, len(cb.Entries), cb.Entries[item.patchIdx].Name)
					}
				}
				if completed%500 == 0 {
					fmt.Printf("  [%d/%d] samples rendered\n", completed, totalSamples)
				}
				mu.Unlock()
			}
		}(w)
	}

	// Now fill the channel (workers are already consuming)
	for _, item := range workItems {
		workChan <- item
	}
	close(workChan)

	wg.Wait()

	// Write labels file sequentially (no race)
	labelsPath := filepath.Join(*outDir, "labels.bin")
	labelsFile, err := os.Create(labelsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating labels file: %v\n", err)
		os.Exit(1)
	}
	defer labelsFile.Close()

	binary.Write(labelsFile, binary.LittleEndian, uint32(totalSamples))
	binary.Write(labelsFile, binary.LittleEndian, uint32(match.DX7LabelDims))
	for _, label := range allLabels {
		binary.Write(labelsFile, binary.LittleEndian, label)
	}

	fmt.Printf("Done: %d total samples rendered -> %s\n", totalSamples, labelsPath)
}

func renderWithEnvelope(synth *msfa.Synth, c *match.Candidate, noteNum int) []int16 {
	patch := c.Encode()
	synth.SetVoice(0, patch)

	synth.WriteMidi([]byte{0x90, byte(noteNum), 100})

	// Render ~2 seconds to capture full envelope including release
	renderLen := 2 * match.RenderSampleRate
	pcm := make([]int16, renderLen)
	synth.GetSamples(pcm)

	// Note Off at 1.5s to capture release tail
	noteOffSample := 3 * match.RenderSampleRate / 2
	synth.WriteMidi([]byte{0x80, byte(noteNum), 0})

	// Render remaining release tail
	remaining := pcm[noteOffSample:]
	synth.GetSamples(remaining)

	return pcm
}

func normalizeTimbreParams(c *match.Candidate) []float32 {
	normalized := make([]float32, match.DX7LabelDims)

	normalized[0] = clampFloat(float32(c.Algorithm) / 31.0)

	for opIdx := 0; opIdx < 6; opIdx++ {
		base := 1 + opIdx*8
		normalized[base+0] = clampFloat(float32(c.Ops[opIdx].Mode))
		normalized[base+1] = clampFloat(float32(c.Ops[opIdx].FreqCoarse) / 31.0)
		normalized[base+2] = clampFloat(float32(c.Ops[opIdx].FreqFine) / 99.0)
		normalized[base+3] = clampFloat(float32(c.Ops[opIdx].OutputLevel) / 99.0)
		normalized[base+4] = clampFloat(float32(c.Ops[opIdx].Rate1) / 99.0)
		normalized[base+5] = clampFloat(float32(c.Ops[opIdx].Rate2) / 99.0)
		normalized[base+6] = clampFloat(float32(c.Ops[opIdx].Level3) / 99.0)
		normalized[base+7] = clampFloat(float32(c.Ops[opIdx].Rate4) / 99.0)
	}

	normalized[49] = clampFloat(float32(c.Feedback) / 7.0)

	// Pitch envelope params (indices 50-52)
	normalized[50] = clampFloat(float32(c.PitchRate1) / 99.0)
	normalized[51] = clampFloat(float32(c.PitchLevel1) / 99.0)
	normalized[52] = clampFloat(float32(c.PitchLevel2) / 99.0)

	return normalized
}

func clampFloat(v float32) float32 {
	if v < 0.0 {
		return 0.0
	}
	if v > 1.0 {
		return 1.0
	}
	return v
}

func saveWav(path string, samples []int16, sampleRate int) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()

	dataSize := len(samples) * 2
	fileSize := 36 + dataSize

	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(fileSize))
	f.Write([]byte("WAVE"))

	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))
	binary.Write(f, binary.LittleEndian, uint16(1))
	binary.Write(f, binary.LittleEndian, uint16(1))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate*2))
	binary.Write(f, binary.LittleEndian, uint16(2))
	binary.Write(f, binary.LittleEndian, uint16(16))

	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(dataSize))

	for _, s := range samples {
		binary.Write(f, binary.LittleEndian, s)
	}
}
