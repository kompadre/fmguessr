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
	variants := flag.Int("variants", 50, "Variants per patch")
	flag.Parse()

	os.MkdirAll(*outDir, 0755)

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
	fmt.Printf("Rendering %d variants per patch at MIDI %v (parallel)...\n", *variants, notes)

	// Build work items: base patch + perturbations × notes
	type workItem struct {
		flatIdx int
		patchIdx int
		variant  int
		note     int
		cand     *match.Candidate
	}

	var workItems []workItem
	rng := rand.New(rand.NewSource(42))
	flatIdx := 0

	for patchIdx, entry := range cb.Entries {
		// Base patch
		base := entry.Candidate.Copy()
		for _, note := range notes {
			workItems = append(workItems, workItem{flatIdx, patchIdx, 0, note, base.Copy()})
			flatIdx++
		}

		// Perturbed variants
		for v := 1; v < *variants; v++ {
			perturbed := perturbCandidate(entry.Candidate, rng)
			for _, note := range notes {
				workItems = append(workItems, workItem{flatIdx, patchIdx, v, note, perturbed.Copy()})
				flatIdx++
			}
		}
	}

	totalSamples := len(workItems)
	fmt.Printf("Total work items: %d (%d patches × %d variants × %d notes)\n",
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
				// Render with natural envelope (not pure sustain)
				// so Python can extract probe features (ADSR, brightness, harmonicity)
				pcm := renderWithEnvelope(workerSynth, item.cand, item.note)

				// Save WAV with note in filename
				wavPath := fmt.Sprintf("%s/p%04d_v%02d_n%03d.wav", *outDir, item.patchIdx, item.variant, item.note)
				saveWav(wavPath, pcm, match.RenderSampleRate)

				// Store label via pre-computed flat index
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
	// Render with natural envelope so Python can extract probe features (ADSR, brightness, etc.)
	patch := c.Encode()
	synth.SetVoice(0, patch)

	// Note On
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

func perturbCandidate(c *match.Candidate, rng *rand.Rand) *match.Candidate {
	d := c.Copy()

	for i := 0; i < 6; i++ {
		// Perturb OutputLevel (σ=6, ±12) with slight attenuation bias
		levelDelta := int(rng.NormFloat64() * 6.0)
		newLevel := int(d.Ops[i].OutputLevel) + levelDelta
		// 30% chance of slight attenuation (reduce by 5-10%)
		if rng.Float64() < 0.30 {
			attenuation := 3 + rng.Intn(4) // 3-6 levels quieter
			newLevel -= attenuation
		}
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

		// Perturb FreqCoarse (σ=1.5, ±3)
		coarseDelta := int(rng.NormFloat64() * 1.5)
		newCoarse := int(d.Ops[i].FreqCoarse) + coarseDelta
		if newCoarse < 0 {
			newCoarse = 0
		}
		if newCoarse > 31 {
			newCoarse = 31
		}
		d.Ops[i].FreqCoarse = uint8(newCoarse)

		// Perturb Rate1/Attack (σ=4, ±8)
		r1Delta := int(rng.NormFloat64() * 4.0)
		newR1 := int(d.Ops[i].Rate1) + r1Delta
		if newR1 < 0 {
			newR1 = 0
		}
		if newR1 > 99 {
			newR1 = 99
		}
		d.Ops[i].Rate1 = uint8(newR1)

		// Perturb Rate2/Decay (σ=4, ±8)
		r2Delta := int(rng.NormFloat64() * 4.0)
		newR2 := int(d.Ops[i].Rate2) + r2Delta
		if newR2 < 0 {
			newR2 = 0
		}
		if newR2 > 99 {
			newR2 = 99
		}
		d.Ops[i].Rate2 = uint8(newR2)

		// Perturb Rate3/Sustain (σ=4, ±8) — move sustain timing
		r3Delta := int(rng.NormFloat64() * 4.0)
		newR3 := int(d.Ops[i].Rate3) + r3Delta
		if newR3 < 0 {
			newR3 = 0
		}
		if newR3 > 99 {
			newR3 = 99
		}
		d.Ops[i].Rate3 = uint8(newR3)

		// Perturb Level3/Sustain (σ=6, ±12)
		l3Delta := int(rng.NormFloat64() * 6.0)
		newL3 := int(d.Ops[i].Level3) + l3Delta
		if newL3 < 0 {
			newL3 = 0
		}
		if newL3 > 99 {
			newL3 = 99
		}
		d.Ops[i].Level3 = uint8(newL3)

		// Perturb Rate4/Release (σ=4, ±8)
		r4Delta := int(rng.NormFloat64() * 4.0)
		newR4 := int(d.Ops[i].Rate4) + r4Delta
		if newR4 < 0 {
			newR4 = 0
		}
		if newR4 > 99 {
			newR4 = 99
		}
		d.Ops[i].Rate4 = uint8(newR4)

		// Perturb Level1/Attack (σ=6, ±12) — move attack endpoint
		l1Delta := int(rng.NormFloat64() * 6.0)
		newL1 := int(d.Ops[i].Level1) + l1Delta
		if newL1 < 0 {
			newL1 = 0
		}
		if newL1 > 99 {
			newL1 = 99
		}
		d.Ops[i].Level1 = uint8(newL1)

		// Perturb Level2/Decay (σ=6, ±12) — move decay endpoint
		l2Delta := int(rng.NormFloat64() * 6.0)
		newL2 := int(d.Ops[i].Level2) + l2Delta
		if newL2 < 0 {
			newL2 = 0
		}
		if newL2 > 99 {
			newL2 = 99
		}
		d.Ops[i].Level2 = uint8(newL2)

		// Perturb Level4/Release (σ=6, ±12) — move release endpoint
		l4Delta := int(rng.NormFloat64() * 6.0)
		newL4 := int(d.Ops[i].Level4) + l4Delta
		if newL4 < 0 {
			newL4 = 0
		}
		if newL4 > 99 {
			newL4 = 99
		}
		d.Ops[i].Level4 = uint8(newL4)
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

	// Perturb pitch envelope (σ=3, 40% chance)
	if rng.Float64() < 0.40 {
		pr1Delta := int(rng.NormFloat64() * 3.0)
		newPR1 := int(d.PitchRate1) + pr1Delta
		if newPR1 < 0 {
			newPR1 = 0
		}
		if newPR1 > 99 {
			newPR1 = 99
		}
		d.PitchRate1 = uint8(newPR1)

		pl1Delta := int(rng.NormFloat64() * 3.0)
		newPL1 := int(d.PitchLevel1) + pl1Delta
		if newPL1 < 0 {
			newPL1 = 0
		}
		if newPL1 > 99 {
			newPL1 = 99
		}
		d.PitchLevel1 = uint8(newPL1)

		pl2Delta := int(rng.NormFloat64() * 3.0)
		newPL2 := int(d.PitchLevel2) + pl2Delta
		if newPL2 < 0 {
			newPL2 = 0
		}
		if newPL2 > 99 {
			newPL2 = 99
		}
		d.PitchLevel2 = uint8(newPL2)
	}

	return d
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
