package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/kompadre/fmguessr/external/miniaudio"
	"github.com/kompadre/fmguessr/external/msfa"
	"github.com/kompadre/fmguessr/pkg/match"
)

const (
	sampleRate  = 44100
	bufferSize  = 8192  // ~186ms ring buffer
	blockSize   = 512   // render block size
)

var noteNames = map[string]int{
	"C": 0, "C#": 1, "Db": 1,
	"D": 2, "D#": 3, "Eb": 3,
	"E": 4,
	"F": 5, "F#": 6, "Gb": 6,
	"G": 7, "G#": 8, "Ab": 8,
	"A": 9, "A#": 10, "Bb": 10,
	"B": 11,
}

func main() {
	outPath := flag.String("o", "", "export to WAV file instead of playing")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <patch.syx|codebook.bin|patches.npy> [slot] [note]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Play an instrument from a DX7 .syx file, codebook .bin, or .npy patch bank.\n")
		fmt.Fprintf(os.Stderr, "  slot: patch slot 0-31 for .syx, 0-N for .bin/.npy (default: play all)\n")
		fmt.Fprintf(os.Stderr, "  note: C3, D#4, Bb5, etc. (default: C3)\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}
	patchPath := flag.Arg(0)

	slot := -1 // -1 = play all
	if flag.NArg() >= 2 {
		s, err := strconv.Atoi(flag.Arg(1))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid slot %q: %v\n", flag.Arg(1), err)
			os.Exit(1)
		}
		slot = s
	}

	noteName := "C3"
	if flag.NArg() >= 3 {
		noteName = flag.Arg(2)
	}

	midiNote, err := parseNote(noteName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid note %q: %v\n", noteName, err)
		os.Exit(1)
	}

	// Initialize synth
	synth := msfa.NewSynth(float64(sampleRate))
	defer synth.Free()

	ext := strings.ToLower(filepath.Ext(patchPath))

	switch ext {
	case ".bin":
		// Load codebook (DXCB or DXCT format)
		cb, err := match.LoadCodebookAny(patchPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading codebook %s: %v\n", patchPath, err)
			os.Exit(1)
		}
		numPatches := len(cb.Entries)
		if numPatches == 0 {
			fmt.Fprintf(os.Stderr, "Codebook is empty\n")
			os.Exit(1)
		}
		if slot >= numPatches {
			fmt.Fprintf(os.Stderr, "Slot %d out of range: codebook has %d entries (0-%d)\n", slot, numPatches, numPatches-1)
			os.Exit(1)
		}

		if slot < 0 {
			// Play all codebook entries
			// Load into synth bank (up to 32 at a time for synth compatibility)
			playCount := numPatches
			if playCount > 32 {
				playCount = 32
			}
			for i := 0; i < playCount; i++ {
				synth.SetVoice(i, cb.Entries[i].Candidate.Encode())
			}
			if *outPath != "" {
				renderBankToWav(synth, nil, playCount, midiNote, *outPath)
			} else {
				playBankThroughSpeaker(synth, playCount, midiNote)
			}
			return
		}

		// Single slot
		synth.SetVoice(0, cb.Entries[slot].Candidate.Encode())
		synth.ChangeProgram(0, 0)
		cand := cb.Entries[slot].Candidate
		fmt.Printf("Loaded: codebook entry %d/%d — %s\n", slot, numPatches, cb.Entries[slot].Name)
		fmt.Printf("  Algorithm: %d, Feedback: %d\n", cand.Algorithm+1, cand.Feedback)
		for i := 0; i < 6; i++ {
			op := cand.Ops[i]
			role := "M"
			if match.IsCarrier(cand.Algorithm, i) {
				role = "C"
			}
			fmt.Printf("  Op%d [%s]: Level=%d Fine=%d Coarse=%d L3=%d R2=%d\n",
				i+1, role, op.OutputLevel, op.FreqFine, op.FreqCoarse, op.Level3, op.Rate2)
		}

	case ".npy":
		// Load patches from .npy file (N×128 uint8 array)
		patchData, numPatches, err := match.LoadPatchBytes(patchPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", patchPath, err)
			os.Exit(1)
		}
		if slot >= numPatches {
			fmt.Fprintf(os.Stderr, "Slot %d out of range: .npy has %d patches (0-%d)\n", slot, numPatches, numPatches-1)
			os.Exit(1)
		}
		// Load up to 32 patches into the synth bank (or fewer if npy has less)
		loadCount := numPatches
		if loadCount > 32 {
			loadCount = 32
		}
		for i := 0; i < loadCount; i++ {
			synth.SetVoice(i, patchData[i*128:(i+1)*128])
		}

		if slot < 0 {
			// Play all patches
			if *outPath != "" {
				renderBankToWav(synth, patchData, loadCount, midiNote, *outPath)
			} else {
				playBankThroughSpeaker(synth, loadCount, midiNote)
			}
			return
		}

		synth.ChangeProgram(0, uint8(slot))
		voiceData := patchData[slot*128 : (slot+1)*128]
		cand := match.DecodeVoicePatch(voiceData)
		if cand != nil {
			fmt.Printf("Loaded: patch %d/%d from %s\n", slot, numPatches, patchPath)
			fmt.Printf("  Algorithm: %d, Feedback: %d\n", cand.Algorithm+1, cand.Feedback)
			for i := 0; i < 6; i++ {
				op := cand.Ops[i]
				role := "M"
				if match.IsCarrier(cand.Algorithm, i) {
					role = "C"
				}
				fmt.Printf("  Op%d [%s]: Level=%d Fine=%d Coarse=%d L3=%d R2=%d\n",
					i+1, role, op.OutputLevel, op.FreqFine, op.FreqCoarse, op.Level3, op.Rate2)
			}
		}

	default:
		// Load .syx file (loads all 32 voices into the bank)
		if slot > 31 {
			fmt.Fprintf(os.Stderr, "Invalid slot %d: must be 0-31 for .syx files\n", slot)
			os.Exit(1)
		}
		if err := synth.LoadPatch(patchPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", patchPath, err)
			os.Exit(1)
		}

		if slot < 0 {
			// Play all 32 patches
			if *outPath != "" {
				renderBankToWav(synth, nil, 32, midiNote, *outPath)
			} else {
				playBankThroughSpeaker(synth, 32, midiNote)
			}
			return
		}

		synth.ChangeProgram(0, uint8(slot))
		fmt.Printf("Loaded: slot %d — %s\n", slot, synth.Programs[slot])

		// Extract and display patch parameters
		bank := make([]byte, 4096)
		synth.GetBank(bank)
		voiceStart := slot * 128
		if voiceStart+128 <= len(bank) {
			cand := match.DecodeVoicePatch(bank[voiceStart : voiceStart+128])
			if cand != nil {
				fmt.Printf("  Algorithm: %d, Feedback: %d\n", cand.Algorithm+1, cand.Feedback)
				for i := 0; i < 6; i++ {
					op := cand.Ops[i]
					role := "M"
					if match.IsCarrier(cand.Algorithm, i) {
						role = "C"
					}
					fmt.Printf("  Op%d [%s]: Level=%d Fine=%d Coarse=%d L3=%d R2=%d\n",
						i+1, role, op.OutputLevel, op.FreqFine, op.FreqCoarse, op.Level3, op.Rate2)
				}
			}
		}
	}

	if *outPath != "" {
		renderToWav(synth, midiNote, *outPath)
	} else {
		playThroughSpeaker(synth, midiNote)
	}
}

func playThroughSpeaker(synth *msfa.Synth, midiNote uint8) {
	err := miniaudio.Init(sampleRate, bufferSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening audio device: %v\n", err)
		os.Exit(1)
	}
	defer miniaudio.Close()

	sustainSamples := int(float64(sampleRate) * 2) // 2 seconds

	left := make([]float32, blockSize)
	right := make([]float32, blockSize)
	pcm := make([]int16, blockSize)

	// Pre-fill ring buffer with silence so the callback never starves
	silence := make([]float32, bufferSize)
	miniaudio.WriteChannels(silence, silence)

	// Note On
	synth.WriteMidi([]byte{0x90, midiNote, 100})
	fmt.Printf("Note On: %d (velocity 100)\n", midiNote)

	// Render sustain period
	rendered := 0
	for rendered < sustainSamples {
		n := blockSize
		if rendered+n > sustainSamples {
			n = sustainSamples - rendered
		}

		// Wait for enough space
		for {
			if miniaudio.AvailableWriteSpace() >= n {
				break
			}
			time.Sleep(time.Millisecond)
		}

		synth.GetSamples(pcm[:n])
		for i := 0; i < n; i++ {
			f := float32(pcm[i]) / 32768.0
			left[i] = f
			right[i] = f
		}
		miniaudio.WriteChannels(left[:n], right[:n])
		rendered += n
	}

	// Note Off
	synth.WriteMidi([]byte{0x80, midiNote, 0})
	fmt.Println("Note Off")

	// Drain: render release tail until silence or hard timeout (3 seconds max)
	drainMax := sampleRate * 3
	for i := 0; i < drainMax; i += blockSize {
		n := blockSize
		if i+n > drainMax {
			n = drainMax - i
		}

		synth.GetSamples(pcm[:n])
		allZero := true
		for j := 0; j < n; j++ {
			if pcm[j] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			break
		}

		for {
			if miniaudio.AvailableWriteSpace() >= n {
				break
			}
			time.Sleep(time.Millisecond)
		}

		for j := 0; j < n; j++ {
			f := float32(pcm[j]) / 32768.0
			left[j] = f
			right[j] = f
		}
		miniaudio.WriteChannels(left[:n], right[:n])
	}

	time.Sleep(100 * time.Millisecond)
	fmt.Println("Done")
}

func renderToWav(synth *msfa.Synth, midiNote uint8, outPath string) {
	// Render sustain + release tail
	sustainDuration := 2 * time.Second
	releaseDuration := 3 * time.Second
	totalDuration := sustainDuration + releaseDuration
	totalSamples := int(float64(sampleRate) * totalDuration.Seconds())

	// Note On
	synth.WriteMidi([]byte{0x90, midiNote, 100})
	fmt.Printf("Note On: %d\n", midiNote)

	// Render all samples into a buffer
	pcm := make([]int16, blockSize)
	samples := make([]int16, 0, totalSamples)
	sustainSamples := int(float64(sampleRate) * sustainDuration.Seconds())
	rendered := 0

	for rendered < totalSamples {
		synth.GetSamples(pcm)
		samples = append(samples, pcm[:]...)
		rendered += blockSize

		// Send Note Off after sustain period
		if rendered >= sustainSamples && rendered < sustainSamples+blockSize {
			synth.WriteMidi([]byte{0x80, midiNote, 0})
			fmt.Println("Note Off")
		}
	}

	// Trim trailing silence
	for len(samples) > 0 && samples[len(samples)-1] == 0 {
		samples = samples[:len(samples)-1]
	}
	// Keep at least the sustain portion
	minSamples := sustainSamples
	if len(samples) < minSamples {
		minSamples = len(samples)
	}
	// Allow some release tail (up to 1 second beyond sustain)
	maxLen := sustainSamples + sampleRate
	if len(samples) > maxLen {
		samples = samples[:maxLen]
	} else if len(samples) < minSamples {
		samples = samples[:minSamples]
	}

	// Write WAV file
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", outPath, err)
		os.Exit(1)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, 1, 1) // 1 = PCM

	buf := &audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
		Data:   make([]int, len(samples)),
	}
	for i, s := range samples {
		buf.Data[i] = int(s)
	}

	if err := enc.Write(buf); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing WAV: %v\n", err)
		os.Exit(1)
	}
	if err := enc.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error closing WAV: %v\n", err)
		os.Exit(1)
	}

	duration := float64(len(samples)) / float64(sampleRate)
	fmt.Printf("Wrote %s (%d samples, %.2fs)\n", outPath, len(samples), duration)
}

// playBankThroughSpeaker plays all patches in the bank sequentially.
// 3s per patch (2s sustain + 1s release), 1s silence between patches.
func playBankThroughSpeaker(synth *msfa.Synth, count int, midiNote uint8) {
	err := miniaudio.Init(sampleRate, bufferSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening audio device: %v\n", err)
		os.Exit(1)
	}
	defer miniaudio.Close()

	bank := make([]byte, 4096)
	synth.GetBank(bank)

	sustainSamples := 2 * sampleRate  // 2s sustain
	releaseSamples := 1 * sampleRate  // 1s release tail
	pauseSamples := 1 * sampleRate    // 1s silence between patches

	left := make([]float32, blockSize)
	right := make([]float32, blockSize)
	pcm := make([]int16, blockSize)

	// Pre-fill ring buffer with silence
	silence := make([]float32, bufferSize)
	miniaudio.WriteChannels(silence, silence)

	for slot := 0; slot < count; slot++ {
		synth.ChangeProgram(0, uint8(slot))

		// Print patch info
		voiceStart := slot * 128
		name := "?"
		if slot < len(synth.Programs) {
			name = synth.Programs[slot]
		}
		if voiceStart+128 <= len(bank) {
			cand := match.DecodeVoicePatch(bank[voiceStart : voiceStart+128])
			if cand != nil {
				fmt.Printf("[%2d] %s  algo=%d fb=%d\n", slot, name, cand.Algorithm+1, cand.Feedback)
			}
		}

		// Note On
		synth.WriteMidi([]byte{0x90, midiNote, 100})

		// Render sustain + release, paced to real-time
		totalSamples := sustainSamples + releaseSamples
		rendered := 0
		for rendered < totalSamples {
			n := blockSize
			if rendered+n > totalSamples {
				n = totalSamples - rendered
			}

			// Wait for enough space
			for {
				avail := miniaudio.AvailableWriteSpace()
				if avail >= n {
					break
				}
				time.Sleep(time.Millisecond)
			}

			synth.GetSamples(pcm[:n])
			for i := 0; i < n; i++ {
				f := float32(pcm[i]) / 32768.0
				left[i] = f
				right[i] = f
			}
			miniaudio.WriteChannels(left[:n], right[:n])
			rendered += n

			// Note Off after sustain
			if rendered >= sustainSamples && rendered < sustainSamples+blockSize {
				synth.WriteMidi([]byte{0x80, midiNote, 0})
			}
		}

		// Silence pause (1s), also paced
		silenceL := make([]float32, blockSize)
		silenceR := make([]float32, blockSize)
		for i := 0; i < pauseSamples; i += blockSize {
			n := blockSize
			if i+n > pauseSamples {
				n = pauseSamples - i
			}
			for {
				avail := miniaudio.AvailableWriteSpace()
				if avail >= n {
					break
				}
				time.Sleep(time.Millisecond)
			}
			miniaudio.WriteChannels(silenceL[:n], silenceR[:n])
		}
	}

	fmt.Println("Done — played all patches")
}

// renderBankToWav renders all patches to a single WAV file.
// 3s per patch (2s sustain + 1s release), 1s silence between patches.
func renderBankToWav(synth *msfa.Synth, patchData []byte, count int, midiNote uint8, outPath string) {
	perPatchSamples := 3 * sampleRate  // 3s per patch
	pauseSamples := 1 * sampleRate     // 1s silence
	totalPerPatch := perPatchSamples + pauseSamples
	totalSamples := totalPerPatch * count

	bank := make([]byte, 4096)
	synth.GetBank(bank)

	fmt.Printf("Rendering %d patches to %s (%.0fs total)...\n", count, outPath, float64(totalSamples)/float64(sampleRate))

	allSamples := make([]int16, 0, totalSamples)
	pcm := make([]int16, blockSize)

	for slot := 0; slot < count; slot++ {
		synth.ChangeProgram(0, uint8(slot))

		// Print patch info
		voiceStart := slot * 128
		name := "?"
		if slot < len(synth.Programs) {
			name = synth.Programs[slot]
		}
		if voiceStart+128 <= len(bank) {
			cand := match.DecodeVoicePatch(bank[voiceStart : voiceStart+128])
			if cand != nil {
				fmt.Printf("  [%2d] %s  algo=%d fb=%d\n", slot, name, cand.Algorithm+1, cand.Feedback)
			}
		}

		// Note On
		synth.WriteMidi([]byte{0x90, midiNote, 100})

		// Render sustain (2s) + release (1s)
		sustainSamples := 2 * sampleRate
		rendered := 0
		for rendered < perPatchSamples {
			synth.GetSamples(pcm)
			allSamples = append(allSamples, pcm[:]...)
			rendered += blockSize

			// Note Off after sustain
			if rendered >= sustainSamples && rendered < sustainSamples+blockSize {
				synth.WriteMidi([]byte{0x80, midiNote, 0})
			}
		}

		// Silence pause
		silenceBlock := make([]int16, blockSize)
		for i := 0; i < pauseSamples; i += blockSize {
			allSamples = append(allSamples, silenceBlock...)
		}
	}

	// Write WAV
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", outPath, err)
		os.Exit(1)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, 1, 1)
	buf := &audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: sampleRate},
		Data:   make([]int, len(allSamples)),
	}
	for i, s := range allSamples {
		buf.Data[i] = int(s)
	}

	if err := enc.Write(buf); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing WAV: %v\n", err)
		os.Exit(1)
	}
	if err := enc.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error closing WAV: %v\n", err)
		os.Exit(1)
	}

	duration := float64(len(allSamples)) / float64(sampleRate)
	fmt.Printf("Wrote %s (%d samples, %.1fs, %d patches)\n", outPath, len(allSamples), duration, count)
}

func parseNote(s string) (uint8, error) {
	s = strings.TrimSpace(s)
	if len(s) < 2 {
		return 0, fmt.Errorf("note too short: %q", s)
	}

	// Extract note name (may be 1 or 2 chars)
	var name string
	var rest string
	if len(s) >= 2 && (s[1] == '#' || s[1] == 'b') {
		name = s[:2]
		rest = s[2:]
	} else {
		name = s[:1]
		rest = s[1:]
	}

	semitone, ok := noteNames[name]
	if !ok {
		return 0, fmt.Errorf("unknown note name: %q", name)
	}

	if len(rest) == 0 {
		return 0, fmt.Errorf("missing octave number: %q", s)
	}

	octave := 0
	for _, c := range rest {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("invalid octave: %q", rest)
		}
		octave = octave*10 + int(c-'0')
	}

	midi := (octave+1)*12 + semitone
	if midi < 0 || midi > 127 {
		return 0, fmt.Errorf("MIDI note %d out of range (0-127)", midi)
	}

	return uint8(midi), nil
}
