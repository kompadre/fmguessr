package main

import (
	"context"
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
	beam := flag.Bool("beam", false, "enable constructive beam search (experimental)")
	beamWidth := flag.Int("beam-width", 4, "beam search width (K best candidates per step)")
	beamRenders := flag.Int("beam-renders", 20000, "beam search render budget")
	cspoa := flag.Bool("cspoa", false, "enable CSPOA progressive attachment (explorative)")
	cspoaBudget := flag.Int("cspoa-budget", 5000, "CSPOA render budget")
	classifier := flag.String("classifier", "", "path to patch classifier ONNX model (optional)")
	classifierOnly := flag.Bool("classifier-only", false, "skip GA, return best classifier predictions only")
	budgetSeconds := flag.Float64("budget-seconds", 0, "stop matching after N seconds (0=unlimited)")
	encoderPath := flag.String("encoder", "", "path to 1D ResNet ONNX encoder model for trajectory distance (optional)")
	sessionOut := flag.String("session-out", "", "save trajectory session to file (optional, for debugging/reuse)")
	sessionIn := flag.String("session-in", "", "load trajectory session from file (skip recomputation)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <target.wav>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Find DX7 FM parameters that produce a sound similar to the target WAV.\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}

	// Go's flag package stops at the first non-flag argument.
	// This means ./match key.wav -syx out.syx silently ignores -syx.
	// Fix: pull all flag-like args to the front before parsing.
	args := os.Args[1:]
	var flags []string
	var positional []string
	for i := 0; i < len(args); i++ {
		if len(args[i]) > 0 && args[i][0] == '-' {
			flags = append(flags, args[i])
			// Consume next arg as value only if this is NOT a boolean flag
			// and the next arg looks like a value (not another flag)
			if i+1 < len(args) && !isBoolFlag(args[i]) &&
				(len(args[i+1]) == 0 || args[i+1][0] != '-') {
				flags = append(flags, args[i+1])
				i++
			}
		} else {
			positional = append(positional, args[i])
		}
	}
	os.Args = append([]string{os.Args[0]}, append(flags, positional...)...)
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

	// Set up trajectory distance if encoder or session file provided
	var trajectorySpec *match.TrajectorySpec
	var lfoSpectrum []float64
	var phaseBounds match.PhaseBounds
	var encoder *match.AudioEncoder

	if *sessionIn != "" {
		// Load pre-computed trajectory from session file
		session, err := match.LoadSession(*sessionIn)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not load session file: %v\n", err)
		} else {
			trajectorySpec = session.Trajectory
			lfoSpectrum = session.LFOSpectrum
			phaseBounds = session.PhaseBounds
			fmt.Printf("  Loaded trajectory session: %d timesteps, %d-dim embeddings\n",
				trajectorySpec.TimeSteps, trajectorySpec.EmbDim)
		}
	} else if *encoderPath != "" {
		// Compute trajectory from encoder model
		var err error
		encoder, err = match.NewAudioEncoder(*encoderPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not load encoder model: %v\n", err)
		} else {
			defer encoder.Close()
			trajectorySpec = match.ExtractTrajectory(encoder, target)
			phaseBounds = match.DetectPhaseBounds(target, sampleRate)
			if trajectorySpec != nil {
				lfoSpectrum = match.ComputeLFOSpectrum(trajectorySpec)
				fmt.Printf("  Trajectory: %d timesteps, %d-dim embeddings, phase=[%d,%d]\n",
					trajectorySpec.TimeSteps, trajectorySpec.EmbDim, phaseBounds.AttackEnd, phaseBounds.SustainEnd)
			}
		}
	}

	// Save session file if requested
	if *sessionOut != "" && trajectorySpec != nil {
		if err := match.SaveSession(*sessionOut, &match.SessionData{
			AudioHash:   match.AudioHash(target),
			Trajectory:  trajectorySpec,
			PhaseBounds: phaseBounds,
			LFOSpectrum: lfoSpectrum,
		}); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not save session file: %v\n", err)
		} else {
			fmt.Printf("  Saved trajectory session to: %s\n", *sessionOut)
		}
	}

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

	// Set up context with budget
	var ctx context.Context
	if *budgetSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), time.Duration(*budgetSeconds*float64(time.Second)))
		defer cancel()
	} else {
		ctx = context.Background()
	}

	// Run optimization
	cfg := match.Config{
		RenderDur:       *duration,
		MIDINote:        uint8(*note),
		NoteFreq:        noteFreq,
		CodebookPath:    *codebookPath,
		ModelPath:       *modelPath,
		Verbose:         *verbose,
		GA:              gaCfg,
		UseBeamSearch:   *beam,
		BeamWidth:       *beamWidth,
		BeamMaxRenders:  *beamRenders,
		CSPOA: match.CSPOAConfig{
			Enabled:    *cspoa,
			MaxRenders: *cspoaBudget,
		},
		ClassifierPath:  *classifier,
		ClassifierOnly:  *classifierOnly,
		Ctx:             ctx,
		Encoder:           encoder,
		TargetTrajectory:  trajectorySpec,
		TargetLFOSpectrum: lfoSpectrum,
		PhaseBounds:       phaseBounds,
		TrajectoryAlphaStart: 0.1,
		TrajectoryAlphaEnd:   0.8,
		TrajectoryConfig:     match.DefaultTrajectoryConfig(),
	}

	fmt.Printf("\nStarting GA: %d generations × %d population = %d evaluations max\n",
		gaCfg.Generations, gaCfg.Population, gaCfg.Generations*gaCfg.Population)
	fmt.Printf("  MIDI note: %d (%s), Render: %.2fs\n", cfg.MIDINote, match.MidiNoteName(int(cfg.MIDINote)), cfg.RenderDur.Seconds())
	fmt.Println("---")

	result := match.Run(target, features, cfg)

	fmt.Println("===")
	fmt.Printf("Best match found (distance=%.6f, %d evaluations):\n", result.BestDist, result.Iters)
	fmt.Println(result.Best.String())

	// Print Timbral Top 5
	if len(result.TimbralTop5) > 0 {
		fmt.Printf("\n--- Timbral Top %d (by static timbre distance) ---\n", len(result.TimbralTop5))
		for i, c := range result.TimbralTop5 {
			fmt.Printf("  #%d: %s\n", i+1, c.String())
		}
	}

	// Print Full Top 5
	if len(result.FullTop5) > 0 {
		fmt.Printf("\n--- Full Top %d (by temporal distance, DiffOptimizer-refined) ---\n", len(result.FullTop5))
		for i, c := range result.FullTop5 {
			fmt.Printf("  #%d: %s\n", i+1, c.String())
		}
	}

	// Print Formant Top 5
	if len(result.FormantTop5) > 0 {
		fmt.Printf("\n--- Formant Top %d (formant-constrained refinement) ---\n", len(result.FormantTop5))
		for i, c := range result.FormantTop5 {
			fmt.Printf("  #%d: %s\n", i+1, c.String())
		}
	}

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

// isBoolFlag returns true if the flag is a boolean (no value) flag.
func isBoolFlag(arg string) bool {
	name := arg
	for len(name) > 0 && name[0] == '-' {
		name = name[1:]
	}
	boolFlags := map[string]bool{
		"v":               true,
		"auto-index":      true,
		"beam":            true,
		"cspoa":           true,
		"classifier-only": true,
	}
	return boolFlags[name]
}
