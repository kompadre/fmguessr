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

const dumpMIDINote = 48 // C3

func main() {
	codebookPath := flag.String("codebook", "dx7_codebook.bin", "Path to codebook binary")
	outputPath := flag.String("output", "dataset_cls.bin", "Output dataset file")
	noteMin := flag.Int("note-min", 36, "Minimum MIDI note (C2)")
	noteMax := flag.Int("note-max", 72, "Maximum MIDI note (C5)")
	renderDur := flag.Duration("dur", 3*time.Second, "Render duration per patch (need ≥3s for full LFO cycles)")
	dumpWav := flag.String("dump-wav", "", "Dump a single rendered patch as WAV and exit")
	dumpPatch := flag.Int("dump-patch", 0, "Codebook index to dump with --dump-wav or --dump-mel")
	dumpMel := flag.Bool("dump-mel", false, "Print mel spectrogram stats for --dump-patch and exit")
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

	// Dump single patch as WAV and exit
	if *dumpWav != "" {
		if *dumpPatch < 0 || *dumpPatch >= numPatches {
			fmt.Fprintf(os.Stderr, "Error: --dump-patch %d out of range (0-%d)\n", *dumpPatch, numPatches-1)
			os.Exit(1)
		}
		entry := cb.Entries[*dumpPatch]
		fmt.Printf("Dumping patch %d (%q) at MIDI %d, dur=%v\n", *dumpPatch, entry.Name, dumpMIDINote, *renderDur)

		synth := msfa.NewSynth(float64(match.RenderSampleRate))
		defer synth.Free()

		// Save normalized (classifier path)
		audio := renderWithSynth(synth, entry.Candidate, dumpMIDINote, *renderDur, true)
		if err := saveWavFloat64(*dumpWav, audio, match.RenderSampleRate); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving WAV: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Saved %s (%d samples, %.2fs) [normalized]\n", *dumpWav, len(audio), float64(len(audio))/float64(match.RenderSampleRate))

		// Save raw (pre-normalization) on a fresh synth to avoid extra_buf contamination
		rawPath := *dumpWav
		if len(rawPath) > 4 && rawPath[len(rawPath)-4:] == ".wav" {
			rawPath = rawPath[:len(rawPath)-4] + "_raw.wav"
		} else {
			rawPath = rawPath + "_raw"
		}
		synthRaw := msfa.NewSynth(float64(match.RenderSampleRate))
		defer synthRaw.Free()
		audioRaw := renderWithSynth(synthRaw, entry.Candidate, dumpMIDINote, *renderDur, false)
		if err := saveWavFloat64(rawPath, audioRaw, match.RenderSampleRate); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving raw WAV: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Saved %s (%d samples, %.2fs) [raw]\n", rawPath, len(audioRaw), float64(len(audioRaw))/float64(match.RenderSampleRate))
		return
	}

	// Dump mel spectrogram for a single patch
	if *dumpMel {
		if *dumpPatch < 0 || *dumpPatch >= numPatches {
			fmt.Fprintf(os.Stderr, "Error: --dump-patch %d out of range (0-%d)\n", *dumpPatch, numPatches-1)
			os.Exit(1)
		}
		entry := cb.Entries[*dumpPatch]
		fmt.Printf("Patch %d: %q\n", *dumpPatch, entry.Name)
		fmt.Printf("LFO bytes (112-116): %x\n", entry.Candidate.RawBase[112:117])
		fmt.Printf("MIDI note: %d, dur: %v\n\n", dumpMIDINote, *renderDur)

		synth := msfa.NewSynth(float64(match.RenderSampleRate))
		defer synth.Free()
		audio := renderWithSynth(synth, entry.Candidate, dumpMIDINote, *renderDur, true)
		mel := match.ComputeMelSpectrogram(audio, match.RenderSampleRate)

		// Print mel spectrogram stats per row (frequency band)
		fmt.Println("Mel spectrogram (128×128) row stats:")
		for i := 0; i < match.MelSpecSize; i++ {
			sum := 0.0
			minV := mel[i][0]
			maxV := mel[i][0]
			for j := 0; j < match.MelSpecSize; j++ {
				sum += mel[i][j]
				if mel[i][j] < minV { minV = mel[i][j] }
				if mel[i][j] > maxV { maxV = mel[i][j] }
			}
			mean := sum / float64(match.MelSpecSize)
			sumSq := 0.0
			for j := 0; j < match.MelSpecSize; j++ {
				d := mel[i][j] - mean
				sumSq += d * d
			}
			stddev := math.Sqrt(sumSq / float64(match.MelSpecSize))
			fmt.Printf("  [%3d] mean=%.4f std=%.4f min=%.4f max=%.4f\n", i, mean, stddev, minV, maxV)
		}
		return
	}

	fmt.Printf("MIDI notes: %d (%d-%d)\n", numNotes, *noteMin, *noteMax)
	fmt.Printf("Total samples: %d\n", totalSamples)
	fmt.Printf("Feature: mel spectrogram %d×%d = %d dims\n", match.MelSpecSize, match.MelSpecSize, match.MelSpecFeatureDim)

	// Pre-allocate output buffer
	features := make([]float32, totalSamples*match.MelSpecFeatureDim)
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
				// Render patch (normalized for mel spectrogram consistency)
				audio := renderWithSynth(synth, item.entry.Candidate, uint8(item.midiNote), *renderDur, true)

				// Extract 128×128 mel spectrogram
				mel := match.ComputeMelSpectrogram(audio, match.RenderSampleRate)

				// Flatten to 1D and store
				offset := item.sampleIdx * match.MelSpecFeatureDim
				for i := 0; i < match.MelSpecSize; i++ {
					for j := 0; j < match.MelSpecSize; j++ {
						features[offset+i*match.MelSpecSize+j] = float32(mel[i][j])
					}
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
	binary.Write(f, binary.LittleEndian, uint32(match.MelSpecFeatureDim))
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
	fmt.Printf("Dataset saved: %d samples, %d features (%d×%d mel), %d patches\n",
		totalSamples, match.MelSpecFeatureDim, match.MelSpecSize, match.MelSpecSize, numPatches)
	fmt.Printf("File size: %.1f MB\n", float64(fileSize)/(1024*1024))
}

// renderWithSynth renders a candidate patch using a pre-created synth instance.
// If normalize is true, peak-normalizes the output to [-1, 1].
func renderWithSynth(synth *msfa.Synth, c *match.Candidate, noteNum uint8, dur time.Duration, normalize bool) []float64 {
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

	if normalize {
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
	}

	return total
}

func saveWavFloat64(path string, samples []float64, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

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
	binary.Write(f, binary.LittleEndian, uint16(2))            // block align
	binary.Write(f, binary.LittleEndian, uint16(16))           // bits per sample
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, uint32(dataSize))

	for _, s := range samples {
		v := s
		if v > 1.0 {
			v = 1.0
		}
		if v < -1.0 {
			v = -1.0
		}
		binary.Write(f, binary.LittleEndian, int16(v*32767))
	}
	return nil
}
