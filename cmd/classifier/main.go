package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/kompadre/fmguessr/external/msfa"
	"github.com/kompadre/fmguessr/pkg/match"
)

func main() {
	codebookPath := flag.String("codebook", "dx7_codebook.bin", "Path to codebook binary")
	outputPath := flag.String("output", "dataset_cls.bin", "Output dataset file")
	noteMin := flag.Int("note-min", 36, "Minimum MIDI note (C2)")
	noteMax := flag.Int("note-max", 72, "Maximum MIDI note (C5)")
	renderDur := flag.Duration("dur", 1*time.Second, "Render duration per patch")
	flag.Parse()

	// Load codebook
	cb, err := match.LoadCodebookAny(*codebookPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading codebook: %v\n", err)
		os.Exit(1)
	}

	numPatches := len(cb.Entries)
	var notes []int
	for n := *noteMin; n <= *noteMax; n++ {
		notes = append(notes, n)
	}
	numNotes := len(notes)
	totalSamples := numPatches * numNotes

	fmt.Printf("Codebook: %d patches\n", numPatches)
	fmt.Printf("MIDI notes: %d (%d-%d)\n", numNotes, *noteMin, *noteMax)
	fmt.Printf("Total samples: %d\n", totalSamples)
	fmt.Printf("Feature dim: %d\n", match.Feature784Dim)

	// Pre-allocate output buffer
	features := make([]float32, totalSamples*match.Feature784Dim)
	patchIndices := make([]uint32, totalSamples)
	midiKeys := make([]uint32, totalSamples)

	// Render in parallel
	numWorkers := runtime.GOMAXPROCS(0)
	type workItem struct {
		sampleIdx int
		patchIdx  int
		midiNote  int
		entry     match.CodebookEntry
	}

	workChan := make(chan workItem, numWorkers*4)
	var wg sync.Mutex
	completed := 0

	// Start workers
	var wgDone sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wgDone.Add(1)
		go func() {
			defer wgDone.Done()
			synth := msfa.NewSynth(float64(match.RenderSampleRate))
			defer synth.Free()

			for item := range workChan {
				// Render patch
				audio := renderWithSynth(synth, item.entry.Candidate, uint8(item.midiNote), *renderDur)

				// Extract 784-dim features
				feat := match.ExtractFeatures784(audio, match.RenderSampleRate, item.midiNote)

				// Store results
				offset := item.sampleIdx * match.Feature784Dim
				for i := 0; i < match.Feature784Dim; i++ {
					features[offset+i] = float32(feat[i])
				}
				patchIndices[item.sampleIdx] = uint32(item.patchIdx)
				midiKeys[item.sampleIdx] = uint32(item.midiNote)

				wg.Lock()
				completed++
				if completed%500 == 0 || completed == totalSamples {
					fmt.Printf("  [%d/%d] samples extracted\n", completed, totalSamples)
				}
				wg.Unlock()
			}
		}()
	}

	// Queue work
	sampleIdx := 0
	for patchIdx, entry := range cb.Entries {
		for _, note := range notes {
			workChan <- workItem{
				sampleIdx: sampleIdx,
				patchIdx:  patchIdx,
				midiNote:  note,
				entry:     entry,
			}
			sampleIdx++
		}
	}
	close(workChan)
	wgDone.Wait()

	// Save binary dataset
	fmt.Printf("Saving dataset to %s...\n", *outputPath)
	f, err := os.Create(*outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, uint32(totalSamples))
	binary.Write(f, binary.LittleEndian, uint32(match.Feature784Dim))
	binary.Write(f, binary.LittleEndian, uint32(numPatches))
	binary.Write(f, binary.LittleEndian, uint32(*noteMin))
	binary.Write(f, binary.LittleEndian, uint32(*noteMax))

	// Patch bytes (128 bytes per patch, for embedded lookup)
	for _, entry := range cb.Entries {
		patch := entry.Candidate.Encode()
		f.Write(patch)
	}

	// Features (float32)
	for _, v := range features {
		binary.Write(f, binary.LittleEndian, v)
	}

	// Labels
	for _, v := range patchIndices {
		binary.Write(f, binary.LittleEndian, v)
	}
	for _, v := range midiKeys {
		binary.Write(f, binary.LittleEndian, v)
	}

	fileSize, _ := f.Seek(0, 2)
	fmt.Printf("Dataset saved: %d samples, %d features, %d patches\n", totalSamples, match.Feature784Dim, numPatches)
	fmt.Printf("File size: %.1f MB\n", float64(fileSize)/(1024*1024))
}

// renderWithSynth renders a candidate patch using a pre-created synth instance.
func renderWithSynth(synth *msfa.Synth, c *match.Candidate, noteNum uint8, dur time.Duration) []float64 {
	patch := c.Encode()
	synth.SetVoice(0, patch)

	synth.WriteMidi([]byte{0x90, noteNum, 100})

	nSamples := int(float64(match.RenderSampleRate) * dur.Seconds())
	pcm := make([]int16, nSamples)
	synth.GetSamples(pcm)

	synth.WriteMidi([]byte{0x80, noteNum, 0})

	drainSamples := match.RenderSampleRate / 4
	drain := make([]int16, drainSamples)
	synth.GetSamples(drain)

	total := make([]float64, nSamples+drainSamples)
	for i := 0; i < nSamples; i++ {
		total[i] = float64(pcm[i]) / 32768.0
	}
	for i := 0; i < drainSamples; i++ {
		total[nSamples+i] = float64(drain[i]) / 32768.0
	}

	// Normalize to [-1, 1]
	maxVal := 0.0
	for _, s := range total {
		if abs := math.Abs(s); abs > maxVal {
			maxVal = abs
		}
	}
	if maxVal > 1e-6 {
		for i := range total {
			total[i] /= maxVal
		}
	}

	return total
}
