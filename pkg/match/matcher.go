package match

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// Config holds optimization parameters.
type Config struct {
	RenderDur    time.Duration
	MIDINote     uint8
	NoteFreq     float64 // User-specified note frequency (0 = use detected F0)
	CodebookPath string  // Path to pre-indexed dx7_codebook.bin (optional)
	ModelPath    string  // Path to ONNX model for NN prediction (optional)
	ModelData    []byte  // Optional: raw ONNX model bytes (overrides ModelPath)
	Verbose      bool
	GA           GAConfig
	ProgressCallback func(percent int, phase string) // Optional callback for API server progress reporting

	// Constructive Beam Search (experimental)
	UseBeamSearch  bool // Enable the 6-step constructive beam builder
	BeamWidth      int  // K best candidates per step (default 4)
	BeamMaxRenders int  // Render budget cap (default 20000)

	// CSPOA: Codebook-Seeded Progressive Operator Attachment (explorative)
	CSPOA CSPOAConfig

	// Patch Classifier: standalone NN that classifies patches from codebook
	ClassifierPath  string // Path to patch classifier ONNX model (optional)
	ClassifierOnly  bool   // Skip GA, return only classifier predictions

	// Ctx for cancellation/budget. If nil, a default context is used.
	// Cancel or set deadline to stop matching early.
	Ctx context.Context

	// Trajectory Distance (optional, requires encoder model).
	// TODO: decouple from Config — pass encoder/trajectory as parameters to Run().
	Encoder           *AudioEncoder     // Pre-loaded encoder instance (shared across calls)
	TargetTrajectory  *TrajectorySpec   // Pre-computed target trajectory
	TargetLFOSpectrum []float64         // Pre-computed target LFO spectrum
	PhaseBounds       PhaseBounds       // Pre-computed phase boundaries
	TrajectoryAlphaStart float64        // Trajectory blend weight at gen 0 (default 0.1)
	TrajectoryAlphaEnd   float64        // Trajectory blend weight at final gen (default 0.8)
	TrajectoryConfig     TrajectoryConfig // Phased weights, LFO params

	// Optimization budget: skip expensive optimization on candidates
	// whose distance exceeds bestKnown * (1 + OptMargin).
	// Default 0.3 = allow optimization up to 30% worse than best.
	OptMargin float64
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		RenderDur: 2 * time.Second,
		MIDINote:  60,
		Verbose:   false,
		GA:        DefaultGAConfig(),
		OptMargin: 0.3,
	}
}

// Result holds the output of the optimization.
type Result struct {
	Best        *Candidate
	BestDist    float64
	Iters       int
	Top         []*Candidate // Top 32 candidates for .syx export
	TimbralTop5 []*Candidate // Top 5 by static timbre distance (deduplicated)
	FullTop5    []*Candidate // Top 5 by temporal distance (deduplicated)
	Favorites   []*Candidate // Candidates appearing in both TimbralTop5 and FullTop5
	FormantTop5 []*Candidate // Top 5 by formant-constrained refinement
}

func reportProgress(cfg Config, percent int, phase string) {
	if cfg.ProgressCallback != nil {
		cfg.ProgressCallback(percent, phase)
	}
}

var runStart time.Time

func logElapsed(cfg Config, format string, args ...interface{}) {
	if cfg.Verbose {
		fmt.Printf("[%dms] "+format, append([]interface{}{logMs()}, args...)...)
	}
}

// logMs returns milliseconds since runStart. Available to all package files.
func logMs() int64 {
	if runStart.IsZero() {
		return 0
	}
	return time.Since(runStart).Milliseconds()
}

// loadedCodebooks holds codebooks loaded once at the start of Run().
// All pipeline stages share these references instead of re-loading from disk.
type loadedCodebooks struct {
	fp     *Codebook        // fingerprint codebook (may be nil)
	timbre *TimbreCodebook  // timbre codebook (may be nil)
}

// loadCodebooksOnce loads both codebooks from disk exactly once.
func loadCodebooksOnce(codebookPath string, verbose bool) loadedCodebooks {
	var cb loadedCodebooks
	if codebookPath == "" {
		return cb
	}

	// Fingerprint codebook
	fp, fpErr := LoadCodebook(codebookPath)
	if fpErr != nil {
		if verbose {
			fmt.Printf("Warning: could not load fingerprint codebook: %v\n", fpErr)
		}
	} else if len(fp.Entries) > 0 {
		cb.fp = fp
		if verbose {
			fmt.Printf("Loaded fingerprint codebook: %d entries\n", len(fp.Entries))
		}
	}

	// Timbre codebook (auto-derived path: *_timbre.bin)
	timbrePath := codebookPath
	if len(codebookPath) > 4 && codebookPath[len(codebookPath)-4:] == ".bin" {
		timbrePath = codebookPath[:len(codebookPath)-4] + "_timbre.bin"
	} else {
		timbrePath = codebookPath + "_timbre.bin"
	}
	if tc, err := LoadTimbreCodebook(timbrePath); err == nil && len(tc.Entries) > 0 {
		cb.timbre = tc
		if verbose {
			fmt.Printf("Loaded timbre codebook: %d entries\n", len(tc.Entries))
		}
	} else if verbose && codebookPath != "" {
		if err != nil {
			fmt.Printf("Note: timbre codebook not loaded: %v\n", err)
		}
	}

	return cb
}

// Run executes the matching pipeline: codebook search → formant generation →
// NN prediction → NN refinement → beam search → CSPOA → classifier → GA →
// hill climbing → formant refinement. Each step is a Matcher that returns
// partial results reconciled into a shared ResultsCohort.
func Run(target []float64, features TargetFeatures, cfg Config) Result {
	runStart = time.Now()

	if cfg.Verbose {
		printFeatures(features)
	}

	cachedTargetMel := ComputeTargetMel(target, RenderSampleRate)

	ctx := cfg.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	cbs := loadCodebooksOnce(cfg.CodebookPath, cfg.Verbose)
	cb := cbs.fp
	timbreCB := cbs.timbre

	cohort := ResultsCohort{}
	pool := CandidatePool{}
	state := MatcherState{}

	steps := []Matcher{
		&CodebookSearch{},
		&FormantGenerate{},
		&NNPredict{},
		&NNRefine{},
		&BeamSearch{},
		&CSPOAStep{},
		&ClassifierStep{},
		&GAStep{},
		&HillClimbStep{},
		&FormantRefineStep{},
	}

	for _, step := range steps {
		reportProgress(cfg, 0, fmt.Sprintf("Running: %s", step.Name()))

		in := MatcherInput{
			Target:    target,
			Features:  features,
			Config:    cfg,
			TargetMel: cachedTargetMel,
			Cohort:    &cohort,
			Pool:      &pool,
			Codebook:  cb,
			TimbreCB:  timbreCB,
			State:     &state,
		}

		result := step.Run(ctx, in)

		if result.Err != nil {
			if cfg.Verbose {
				fmt.Printf("Step %s: %v (skipping)\n", step.Name(), result.Err)
			}
			continue
		}

		cohort.Reconcile(result.Cohort)

		if result.EarlyReturn {
			if cfg.Verbose {
				fmt.Printf("Step %s: early return\n", step.Name())
			}
			break
		}
	}

	return finalize(&cohort, &pool, &state, target, features, cachedTargetMel, cfg)
}

// findFavorites returns candidates that appear in both lists (by encoded patch signature).
func findFavorites(timbral, full []*Candidate) []*Candidate {
	fullSet := make(map[string]bool, len(full))
	for _, c := range full {
		fullSet[string(c.Encode())] = true
	}
	var favs []*Candidate
	for _, c := range timbral {
		if fullSet[string(c.Encode())] {
			favs = append(favs, c)
		}
	}
	return favs
}

// checkExactMatch returns true if any island has found a match <= 0.0001.
func checkExactMatch(islandGAs []*GA) bool {
	for _, ga := range islandGAs {
		if ga != nil && ga.bestEverFit <= 1e-4 {
			return true
		}
	}
	return false
}

func applyStep(c *Candidate, which int, step int8) {
	if which == 0 {
		return // Don't line search algorithm jumps
	}
	if which == NumCandidateParams-1 {
		c.Feedback = clampU8(c.Feedback, step, 0, 7)
		return
	}
	opIdx := (which - 1) / 8
	paramIdx := (which - 1) % 8
	switch paramIdx {
	case 1:
		c.Ops[opIdx].FreqCoarse = clampU8(c.Ops[opIdx].FreqCoarse, step, 0, 31)
	case 2:
		c.Ops[opIdx].FreqFine = clampU8(c.Ops[opIdx].FreqFine, step, 0, 99)
	case 3:
		c.Ops[opIdx].OutputLevel = clampU8(c.Ops[opIdx].OutputLevel, step, 75, 99)
	case 4:
		c.Ops[opIdx].Rate1 = clampU8(c.Ops[opIdx].Rate1, step, 10, 99)
	case 5:
		c.Ops[opIdx].Rate2 = clampU8(c.Ops[opIdx].Rate2, step, 5, 99)
	case 6:
		minL3 := uint8(0)
		if IsCarrier(c.Algorithm, opIdx) {
			minL3 = 50 // Carriers keep audible sustain floor
		}
		c.Ops[opIdx].Level3 = clampU8(c.Ops[opIdx].Level3, step, minL3, 99)
	case 7:
		c.Ops[opIdx].Rate4 = clampU8(c.Ops[opIdx].Rate4, step, 10, 99)
	}
}

// evaluateBatchParallel evaluates candidates concurrently across CPU cores using a multi-threaded worker pool.
func evaluateBatchParallel(candidates []*Candidate, target []float64, note uint8, dur time.Duration, rawTargetRMS float64, targetMel *PrecomputedTargetMel) ([]float64, []TargetFeatures) {
	n := len(candidates)
	results := make([]float64, n)
	features := make([]TargetFeatures, n)

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers < 1 {
		numWorkers = 1
	}

	taskChan := make(chan int, n)
	for i := 0; i < n; i++ {
		taskChan <- i
	}
	close(taskChan)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range taskChan {
				candAudio := RenderCandidate(candidates[idx], note, dur)
				trimmed := TrimOrPad(candAudio, len(target))
				rawCandRMS, candidate := CalcRMSAndNormalize(0.1, trimmed)
				var dist float64
				if targetMel != nil {
					dist = MelCosineDistanceCached(targetMel, candidate, RenderSampleRate)
				} else {
					dist = MelCosineDistance(target, candidate, RenderSampleRate)
				}
				if rawTargetRMS > 1e-4 {
					dist += CalcDBPenalty(rawCandRMS, rawTargetRMS)
				}
				results[idx] = dist
				features[idx] = ExtractBrightness(candidate, RenderSampleRate)
			}
		}()
	}
	wg.Wait()

	return results, features
}

func printFeatures(f TargetFeatures) {
	fmt.Println("--- Target Features ---")
	if f.F0 > 0 {
		fmt.Printf("  F0: %.1f Hz (confidence: %.2f)\n", f.F0, f.F0Confidence)
	}
	fmt.Printf("  Attack: %.1f ms, Decay: %.1f ms, Sustain: %.2f, Release: %.1f ms\n",
		f.AttackMs, f.DecayMs, f.SustainLevel, f.ReleaseMs)
	fmt.Printf("  Brightness: %.2f, Harmonicity: %.2f\n", f.BrightnessScore, f.Harmonicity)
	fmt.Printf("  Spectral centroid: %.0f Hz\n", f.SpectralCentroidHz)
	if len(f.FrequencyRatios) > 0 {
		fmt.Printf("  Frequency ratios: %v\n", f.FrequencyRatios)
	}
	fmt.Println("-----------------------")
}
