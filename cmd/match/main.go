package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func main() {
	duration := flag.Duration("d", 2*time.Second, "render duration per candidate")
	note := flag.Int("note", -1, "MIDI note number (0-127), -1 to auto-detect")
	verbose := flag.Bool("v", false, "print progress and feature analysis")
	syxOut := flag.String("syx", "", "export best match as DX7 .syx file")
	codebookPath := flag.String("codebook", "dx7_codebook.bin", "path to pre-indexed dx7_codebook.bin")
	modelPath := flag.String("model", "", "path to ONNX model for NN prediction (optional)")
	generations := flag.Int("generations", 200, "GA generations")
	population := flag.Int("population", 128, "GA population size (32-256)")
	autoIndex := flag.Bool("auto-index", true, "automatically index best result into codebook")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <target.wav>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Find DX7 FM parameters that produce a sound similar to the target WAV.\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}
	wavPath := flag.Arg(0)

	// Limit requested duration to a maximum of 3 seconds
	maxDuration := 3 * time.Second
	if *duration > maxDuration {
		fmt.Printf("Capping requested render duration from %.2fs to maximum %.2fs\n",
			duration.Seconds(), maxDuration.Seconds())
		*duration = maxDuration
	}

	// Load target WAV
	fmt.Printf("Loading target: %s\n", wavPath)
	target, sampleRate, err := match.LoadWav(wavPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading WAV: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Sample rate: %d Hz, Samples: %d (%.2fs)\n",
		sampleRate, len(target), float64(len(target))/float64(sampleRate))

	// Truncate target WAV to requested duration (max 3 seconds)
	maxTargetSamples := int(float64(sampleRate) * duration.Seconds())
	if len(target) > maxTargetSamples {
		fmt.Printf("  Truncating target WAV from %.2fs to requested duration %.2fs\n",
			float64(len(target))/float64(sampleRate), duration.Seconds())
		target = target[:maxTargetSamples]
	}

	// Extract features on the raw target audio for absolute maximum accuracy and TargetRMS
	features := match.ExtractFeatures(target, sampleRate)

	// Normalize target
	target = match.NormalizeRMS(target)

	// Automatically detect root note from target pitch if requested
	if *note == -1 {
		midiNote := 60
		if features.F0 > 20 && features.F0Confidence > 0.3 {
			midiNote = 69 + int(math.Round(12.0*math.Log2(features.F0/440.0)))
			if midiNote < 0 {
				midiNote = 0
			}
			if midiNote > 127 {
				midiNote = 127
			}
			fmt.Printf("  Auto-detected root note: %s (MIDI %d) from F0 %.1f Hz\n",
				match.MidiNoteName(midiNote), midiNote, features.F0)
		} else {
			fmt.Printf("  Could not reliably detect root note. Defaulting to C4 (MIDI 60)\n")
		}
		*note = midiNote
	} else {
		fmt.Printf("  Using user-specified root note: %s (MIDI %d)\n", match.MidiNoteName(*note), *note)
	}

	// Pad/trim target to exact render duration and normalize comparison window
	targetLen := int(float64(match.RenderSampleRate) * duration.Seconds())
	target = match.TrimOrPad(target, targetLen)
	target = match.NormalizeRMS(target)
	fmt.Printf("  Comparison length: %d samples (%.2fs)\n", len(target), duration.Seconds())

	// Configure GA
	gaCfg := match.DefaultGAConfig()
	gaCfg.Generations = *generations
	if *population > 256 {
		*population = 256
	}
	if *population < 32 {
		*population = 32
	}
	if *population%2 != 0 {
		*population--
	}
	gaCfg.Population = *population

	// Compute note frequency from MIDI note number (A4 = 69 = 440 Hz)
	noteFreq := 440.0 * math.Pow(2, float64(*note-69)/12.0)

	// Run optimization
	cfg := match.Config{
		RenderDur:    *duration,
		MIDINote:     uint8(*note),
		NoteFreq:     noteFreq,
		CodebookPath: *codebookPath,
		ModelPath:    *modelPath,
		Verbose:      *verbose,
		GA:           gaCfg,
	}

	fmt.Printf("\nStarting GA: %d generations × %d population = %d evaluations max\n",
		gaCfg.Generations, gaCfg.Population, gaCfg.Generations*gaCfg.Population)
	fmt.Printf("  MIDI note: %d (%s), Render: %.2fs\n", cfg.MIDINote, match.MidiNoteName(int(cfg.MIDINote)), cfg.RenderDur.Seconds())
	fmt.Println("---")

	result := match.Run(target, features, cfg)

	fmt.Println("===")
	fmt.Printf("Best match found (distance=%.6f, %d evaluations):\n", result.BestDist, result.Iters)
	fmt.Println(result.Best.String())

	// Export .syx if requested
	if *syxOut != "" {
		fmt.Printf("Dumping result top: %v.\n", result.Top)
		syx := match.SysexBulkDumpMultiple(result.Top)
		if err := os.WriteFile(*syxOut, syx, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing .syx: %v\n", err)
			os.Exit(1)
		}
		nVoices := len(result.Top)
		if nVoices > 32 {
			nVoices = 32
		}
		fmt.Printf("Wrote %s (DX7 bulk dump, %d voices from best to %dth best)\n",
			*syxOut, nVoices, nVoices)
	}

	// Auto-index best result into codebook by default
	if *autoIndex && result.Best != nil {
		patchName := filepath.Base(wavPath)
		if ext := filepath.Ext(patchName); ext != "" {
			patchName = patchName[:len(patchName)-len(ext)]
		}
		if len(patchName) > 16 {
			patchName = patchName[:16]
		}
		gateDur := time.Duration(features.GateMs * float64(time.Millisecond))
		if err := match.IndexCandidateToCodebook(result.Best, patchName, uint8(*note), gateDur, *duration, *codebookPath); err != nil {
			fmt.Printf("Warning: failed to auto-index match into codebook: %v\n", err)
		} else {
			fmt.Printf("Auto-indexed best match '%s' into codebook %s\n", patchName, *codebookPath)
		}
	}
}
