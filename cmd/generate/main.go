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
	variants := flag.Int("variants", 50, "Variants per patch")
	flag.Parse()

	os.MkdirAll(*outDir, 0755)

	// Load codebook (supports both standard DXCB and timbre DXCT formats)
	cb, err := match.LoadCodebookAny(*codebookPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading codebook: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d patches from %s\n", len(cb.Entries), *codebookPath)
	fmt.Printf("Rendering %d variants per patch at MIDI %d (parallel)...\n", *variants, *noteNum)

	// Build work items: base patch + perturbations
	type workItem struct {
		patchIdx int
		variant  int
		cand     *match.Candidate
	}

	var workItems []workItem
	rng := rand.New(rand.NewSource(42))

	for patchIdx, entry := range cb.Entries {
		// Base patch
		workItems = append(workItems, workItem{patchIdx, 0, entry.Candidate.Copy()})

		// Perturbed variants
		for v := 1; v < *variants; v++ {
			perturbed := perturbCandidate(entry.Candidate, rng)
			workItems = append(workItems, workItem{patchIdx, v, perturbed})
		}
	}

	totalSamples := len(workItems)
	fmt.Printf("Total work items: %d\n", totalSamples)

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
				// Render pure sustain: skip attack, render sustain-only
				pcm := renderPureSustain(workerSynth, item.cand, *noteNum, *outDir, item.patchIdx, item.variant)

				// Save WAV (each file is unique per patchIdx+variant, no lock needed)
				wavPath := fmt.Sprintf("%s/p%04d_v%02d.wav", *outDir, item.patchIdx, item.variant)
				saveWav(wavPath, pcm, match.RenderSampleRate)

				// Store label (thread-safe index assignment)
				label := normalizeTimbreParams(item.cand)
				allLabels[item.patchIdx**variants+item.variant] = label

				mu.Lock()
				completed++
				patchProgress := (item.patchIdx + 1)
				if item.variant == 0 {
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
	binary.Write(labelsFile, binary.LittleEndian, uint32(match.NumCandidateParams))
	for _, label := range allLabels {
		binary.Write(labelsFile, binary.LittleEndian, label)
	}

	fmt.Printf("Done: %d total samples rendered -> %s\n", totalSamples, labelsPath)
}

func renderPureSustain(synth *msfa.Synth, c *match.Candidate, noteNum int, outDir string, patchIdx, variant int) []int16 {
	// Create an envelope-neutral version of the patch
	// DX7 rates: 0=slowest, 99=fastest
	// Levels: 0=silence, 99=max
	// For pure sustain: instant attack, instant decay to sustain, hold indefinitely
	neutral := c.Copy()
	for i := 0; i < 6; i++ {
		neutral.Ops[i].Rate1 = 99  // Instant attack
		neutral.Ops[i].Rate2 = 99  // Instant decay to sustain
		neutral.Ops[i].Level3 = 99 // 100% sustain level
		neutral.Ops[i].Rate4 = 99  // Fast release (unused, no Note Off sent)
	}

	patch := neutral.Encode()
	synth.SetVoice(0, patch)

	// Render 1 second of pure sustained tone
	attackSkip := 50 * match.RenderSampleRate / 1000 // 50ms attack prime
	sustainLen := match.RenderSampleRate              // 1 full second

	// Prime the synth with a short attack burst
	prime := make([]int16, attackSkip)
	synth.WriteMidi([]byte{0x90, byte(noteNum), 100})
	synth.GetSamples(prime)

	// Capture exactly 1 second of stable sustain — NO Note Off
	pcm := make([]int16, sustainLen)
	synth.GetSamples(pcm)

	return pcm
}

func perturbCandidate(c *match.Candidate, rng *rand.Rand) *match.Candidate {
	d := c.Copy()

	for i := 0; i < 6; i++ {
		// Perturb OutputLevel (σ=6, ±12)
		levelDelta := int(rng.NormFloat64() * 6.0)
		newLevel := int(d.Ops[i].OutputLevel) + levelDelta
		if newLevel < 0 {
			newLevel = 0
		}
		if newLevel > 99 {
			newLevel = 99
		}
		d.Ops[i].OutputLevel = uint8(newLevel)

		// Perturb FreqFine (σ=2, ±4)
		fineDelta := int(rng.NormFloat64() * 2.0)
		newFine := int(d.Ops[i].FreqFine) + fineDelta
		if newFine < 0 {
			newFine = 0
		}
		if newFine > 99 {
			newFine = 99
		}
		d.Ops[i].FreqFine = uint8(newFine)
	}

	// Perturb Feedback (σ=1.5, 30% chance)
	if rng.Float64() < 0.30 {
		fbDelta := int(rng.NormFloat64() * 1.5)
		newFb := int(d.Feedback) + fbDelta
		if newFb < 0 {
			newFb = 0
		}
		if newFb > 7 {
			newFb = 7
		}
		d.Feedback = uint8(newFb)
	}

	return d
}

func normalizeTimbreParams(c *match.Candidate) []float32 {
	normalized := make([]float32, match.NumCandidateParams)

	normalized[0] = clampFloat(float32(c.Algorithm) / 31.0)

	for opIdx := 0; opIdx < 6; opIdx++ {
		base := 1 + opIdx*8
		normalized[base+0] = clampFloat(float32(c.Ops[opIdx].Mode))
		normalized[base+1] = clampFloat(float32(c.Ops[opIdx].FreqCoarse) / 31.0)
		normalized[base+2] = clampFloat(float32(c.Ops[opIdx].FreqFine) / 99.0)
		normalized[base+3] = clampFloat(float32(c.Ops[opIdx].OutputLevel) / 99.0)
		normalized[base+4] = 0.0 // Rate1 neutral
		normalized[base+5] = 0.0 // Rate2 neutral
		normalized[base+6] = 1.0 // Level3 = 100% sustain
		normalized[base+7] = 0.0 // Rate4 neutral
	}

	normalized[49] = clampFloat(float32(c.Feedback) / 7.0)

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
