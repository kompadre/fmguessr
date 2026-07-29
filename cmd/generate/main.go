package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

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

	// Load codebook
	cb, err := match.LoadCodebook(*codebookPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading codebook: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d patches from %s\n", len(cb.Entries), *codebookPath)
	fmt.Printf("Rendering %d variants per patch at MIDI %d...\n", *variants, *noteNum)

	// Create synth for rendering
	synth := msfa.NewSynth(float64(match.RenderSampleRate))
	defer synth.Free()

	// Labels file: [n_samples uint32, n_params uint32, labels float32...]
	labelsFile, err := os.Create(filepath.Join(*outDir, "labels.bin"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating labels file: %v\n", err)
		os.Exit(1)
	}
	defer labelsFile.Close()

	// Write placeholder header (update later)
	binary.Write(labelsFile, binary.LittleEndian, uint32(0))
	binary.Write(labelsFile, binary.LittleEndian, uint32(match.NumCandidateParams))

	rng := rand.New(rand.NewSource(42))
	totalSamples := 0

	for patchIdx, entry := range cb.Entries {
		if (patchIdx+1)%100 == 0 || patchIdx == 0 {
			fmt.Printf("  [%d/%d] %s\n", patchIdx+1, len(cb.Entries), entry.Name)
		}

		// Render base patch
		renderPatch(synth, entry.Candidate, *noteNum, *outDir, patchIdx, 0, labelsFile)

		// Create perturbed variants
		for v := 1; v < *variants; v++ {
			perturbed := perturbCandidate(entry.Candidate, rng)
			renderPatch(synth, perturbed, *noteNum, *outDir, patchIdx, v, labelsFile)
			totalSamples++
		}
		totalSamples++ // count base patch
	}

	// Update header with total count
	labelsFile.Seek(0, 0)
	binary.Write(labelsFile, binary.LittleEndian, uint32(totalSamples))

	fmt.Printf("Done: %d total samples rendered\n", totalSamples)
}

func renderPatch(synth *msfa.Synth, c *match.Candidate, noteNum int, outDir string, patchIdx, variant int, labelsFile *os.File) {
	patch := c.Encode()
	synth.SetVoice(0, patch)

	// Render 1 second
	synth.WriteMidi([]byte{0x90, byte(noteNum), 100})
	pcm := make([]int16, match.RenderSampleRate)
	synth.GetSamples(pcm)
	synth.WriteMidi([]byte{0x80, byte(noteNum), 0})

	// Drain
	drain := make([]int16, match.RenderSampleRate/4)
	synth.GetSamples(drain)

	// Save WAV
	wavPath := fmt.Sprintf("%s/p%04d_v%02d.wav", outDir, patchIdx, variant)
	saveWav(wavPath, pcm, match.RenderSampleRate)

	// Write label (50 normalized params)
	label := normalizeParams(c)
	binary.Write(labelsFile, binary.LittleEndian, label)
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

func normalizeParams(c *match.Candidate) []float32 {
	normalized := make([]float32, match.NumCandidateParams)

	// Algorithm
	normalized[0] = float32(c.Algorithm) / 31.0

	// Per-operator params
	for opIdx := 0; opIdx < 6; opIdx++ {
		base := 1 + opIdx*8
		normalized[base+0] = float32(c.Ops[opIdx].Mode)
		normalized[base+1] = float32(c.Ops[opIdx].FreqCoarse) / 31.0
		normalized[base+2] = float32(c.Ops[opIdx].FreqFine) / 99.0
		normalized[base+3] = float32(c.Ops[opIdx].OutputLevel) / 99.0
		normalized[base+4] = float32(c.Ops[opIdx].Rate1) / 99.0
		normalized[base+5] = float32(c.Ops[opIdx].Rate2) / 99.0
		normalized[base+6] = float32(c.Ops[opIdx].Level3) / 99.0
		normalized[base+7] = float32(c.Ops[opIdx].Rate4) / 99.0
	}

	// Feedback
	normalized[49] = float32(c.Feedback) / 7.0

	return normalized
}

func saveWav(path string, samples []int16, sampleRate int) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()

	// WAV header
	dataSize := len(samples) * 2
	fileSize := 36 + dataSize

	f.Write([]byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(fileSize))
	f.Write([]byte("WAVE"))

	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))
	binary.Write(f, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(f, binary.LittleEndian, uint16(1)) // mono
	binary.Write(f, binary.LittleEndian, uint32(sampleRate))
	binary.Write(f, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	binary.Write(f, binary.LittleEndian, uint16(2))             // block align
	binary.Write(f, binary.LittleEndian, uint16(16))            // bits per sample

	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(dataSize))

	for _, s := range samples {
		binary.Write(f, binary.LittleEndian, s)
	}
}
