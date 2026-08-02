package match

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
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

	// Deduplication thresholds for timbral/full top-N selection.
	// Candidates closer than the threshold are considered duplicates; the better one is kept.
	StaticDedupThreshold   float64 // Step 7: timbral dedup by SpectralDistance (default 0.015)
	TemporalDedupThreshold float64 // Step 8: full dedup by TemporalTimbreDistance (default 0.020)
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		RenderDur:             2 * time.Second,
		MIDINote:              60,
		Verbose:               false,
		GA:                    DefaultGAConfig(),
		StaticDedupThreshold:  0.015,
		TemporalDedupThreshold: 0.020,
	}
}

// Result holds the output of the optimization.
type Result struct {
	Best        *Candidate
	BestDist    float64
	Iters       int
	Top         []*Candidate // Top 32 candidates for .syx export
	TimbralTop5 []*Candidate // Top 5 by static timbre distance (deduplicated)
	FullTop5    []*Candidate // Top 5 by temporal distance (DiffOptimizer-refined, deduplicated)
}

func reportProgress(cfg Config, percent int, phase string) {
	if cfg.ProgressCallback != nil {
		cfg.ProgressCallback(percent, phase)
	}
}

// Run executes the genetic algorithm optimization using a 32-Island Niche Model and Multi-Core Parallel Evaluation.
func Run(target []float64, features TargetFeatures, cfg Config) Result {
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	if cfg.Verbose {
		printFeatures(features)
	}

	reportProgress(cfg, 5, "Step 1: Codebook Lookup")

	// Step 1: Codebook Lookup (FIRST step in matching pipeline)
	var knnMap map[uint8]*Candidate
	var codebookPool *CodebookPool
	var bestCodebookCand *Candidate
	bestCodebookDist := math.MaxFloat64

	if cfg.CodebookPath != "" {
		// Load fingerprint codebook
		cb, cbErr := LoadCodebook(cfg.CodebookPath)

		// Load timbre codebook (auto-derived path: *_timbre.bin)
		var timbreCB *TimbreCodebook
		timbreCBPath := cfg.CodebookPath
		if len(cfg.CodebookPath) > 4 && cfg.CodebookPath[len(cfg.CodebookPath)-4:] == ".bin" {
			timbreCBPath = cfg.CodebookPath[:len(cfg.CodebookPath)-4] + "_timbre.bin"
		} else {
			timbreCBPath = cfg.CodebookPath + "_timbre.bin"
		}
		if tc, err := LoadTimbreCodebook(timbreCBPath); err == nil && len(tc.Entries) > 0 {
			timbreCB = tc
			if cfg.Verbose {
				fmt.Printf("Loaded timbre codebook: %d entries from %s\n", len(tc.Entries), timbreCBPath)
			}
		}

		if cbErr == nil && len(cb.Entries) > 0 {
			targetFp := ComputeFingerprint(target, RenderSampleRate)

			// Dual-codebook search when both are available
			if timbreCB != nil && len(timbreCB.Entries) > 0 {
				timbreSpec := ExtractTimbreWindow(target, RenderSampleRate)
				fpMatches, fpDists := cb.SearchKNNWithDistances(targetFp, len(cb.Entries))
				timMatches, timDists := timbreCB.SearchKNNTimbreWithDistances(timbreSpec, len(timbreCB.Entries))

				knnMatches := mergeCodebookResults(fpMatches, fpDists, timMatches, timDists, 48, 0.10)
				if cfg.Verbose {
					fmt.Printf("Dual-codebook: %d FP + %d Timbre → %d merged candidates\n",
						len(fpMatches), len(timMatches), len(knnMatches))
				}

				if len(knnMatches) > 0 {
					knnFits, _ := evaluateBatchParallel(knnMatches, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
					knnMap = make(map[uint8]*Candidate)
					bestDistPerAlgo := make(map[uint8]float64)

					for i, matchCand := range knnMatches {
						algo := matchCand.Algorithm
						dist := knnFits[i]
						if dist < bestCodebookDist {
							bestCodebookDist = dist
							bestCodebookCand = matchCand.Copy()
							bestCodebookCand.Fitness = dist
						}
						bestDist, exists := bestDistPerAlgo[algo]
						if !exists || dist < bestDist {
							bestDistPerAlgo[algo] = dist
							knnMap[algo] = matchCand
						}
					}
					if cfg.Verbose {
						fmt.Printf("Step 1 Dual-Codebook: evaluated %d candidates (best dist=%.6f)\n",
							len(knnMatches), bestCodebookDist)
					}
				}
			} else {
				// Fallback: fingerprint-only search
				topK := len(cb.Entries)
				knnMatches := cb.SearchKNN(targetFp, topK)
				if len(knnMatches) > 0 {
					knnFits, _ := evaluateBatchParallel(knnMatches, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
					knnMap = make(map[uint8]*Candidate)
					bestDistPerAlgo := make(map[uint8]float64)

					for i, matchCand := range knnMatches {
						algo := matchCand.Algorithm
						dist := knnFits[i]
						if dist < bestCodebookDist {
							bestCodebookDist = dist
							bestCodebookCand = matchCand.Copy()
							bestCodebookCand.Fitness = dist
						}
						bestDist, exists := bestDistPerAlgo[algo]
						if !exists || dist < bestDist {
							bestDistPerAlgo[algo] = dist
							knnMap[algo] = matchCand
						}
					}
					if cfg.Verbose {
						fmt.Printf("Step 1 Fingerprint-Only: evaluated %d candidates (best dist=%.6f)\n",
							len(knnMatches), bestCodebookDist)
					}
				}
			}

			// Early return check if codebook match is exact (dist <= 0.0001)
			if bestCodebookDist <= 1e-4 {
				if cfg.Verbose {
					fmt.Printf("Exact codebook match found in Step 1 (dist=%.6f <= 0.0001). Returning early!\n", bestCodebookDist)
				}
				reportProgress(cfg, 100, "Completed (Exact Codebook Match)")
				return Result{
					Best:     bestCodebookCand,
					BestDist: bestCodebookDist,
					Iters:    len(cb.Entries),
					Top:      []*Candidate{bestCodebookCand},
				}
			}

			// Track unused codebook entries for stagnation injection
			usedSigs := make(map[string]bool, len(knnMap))
			for _, cand := range knnMap {
				usedSigs[string(cand.Encode())] = true
			}
			var unusedEntries []*Candidate
			for _, entry := range cb.Entries {
				if !usedSigs[string(entry.Candidate.Encode())] {
					unusedEntries = append(unusedEntries, entry.Candidate)
				}
			}
			if len(unusedEntries) > 0 {
				codebookPool = NewCodebookPool(unusedEntries)
				if cfg.Verbose {
					fmt.Printf("Codebook: %d unused presets available for stagnation injection\n", len(unusedEntries))
				}
			}
		}
	}

	reportProgress(cfg, 10, "Step 2: Timbre & NN Prediction")

	// Phase 0: Extract timbre window from target
	timbreSpec := ExtractTimbreWindow(target, RenderSampleRate)
	if cfg.Verbose {
		fmt.Printf("Extracted timbre window: %d bins\n", TimbreSpecBins)
	}

	// Step 2: Try NN prediction if model available
	var nnCandidates []*Candidate
	if len(cfg.ModelData) > 0 || cfg.ModelPath != "" {
		var predictor *Predictor
		var err error
		if len(cfg.ModelData) > 0 {
			predictor, err = NewPredictorFromBytes(cfg.ModelData)
		} else {
			predictor, err = NewPredictor(cfg.ModelPath)
		}
		if err != nil {
			fmt.Printf("Warning: could not load NN model: %v\n", err)
		} else {
			defer predictor.Close()

			if cfg.Verbose {
				if len(cfg.ModelData) > 0 {
					fmt.Printf("Using NN predictor: embedded (%d bytes)\n", len(cfg.ModelData))
				} else {
					fmt.Printf("Using NN predictor: %s\n", cfg.ModelPath)
				}
			}

			// Load timbre codebook for RawBase attachment
			var timbreCB *TimbreCodebook
			if cfg.CodebookPath != "" {
				timbreCBPath := cfg.CodebookPath
				if len(cfg.CodebookPath) > 4 && cfg.CodebookPath[len(cfg.CodebookPath)-4:] == ".bin" {
					timbreCBPath = cfg.CodebookPath[:len(cfg.CodebookPath)-4] + "_timbre.bin"
				} else {
					timbreCBPath = cfg.CodebookPath + "_timbre.bin"
				}
				if tcb, err := LoadTimbreCodebook(timbreCBPath); err == nil {
					timbreCB = tcb
				}
			}

			// NN predicts continuous params (fines, levels, feedback)
			predicted, nnResult := predictor.PredictToCandidate(timbreSpec)

			// Attach RawBase from nearest timbre codebook entry
			if timbreCB != nil {
				predicted = AttachNearestRawBaseTimbre(predicted, timbreCB)
			} else if cfg.CodebookPath != "" {
				if cb, err := LoadCodebook(cfg.CodebookPath); err == nil {
					predicted = AttachNearestRawBase(predicted, cb)
				}
			}

			if cfg.Verbose {
				fmt.Printf("NN continuous prediction: Feedback=%.2f\n", nnResult.Feedback)
			}

			type algoScore struct {
				algo uint8
				dist float64
				cand *Candidate
			}
			var scores []algoScore

			for algo := uint8(0); algo < 32; algo++ {
				cand := predicted.Copy()
				cand.Algorithm = algo
				if timbreCB != nil {
					cand = AttachNearestRawBaseTimbre(cand, timbreCB)
				} else if cfg.CodebookPath != "" {
					if cb, err := LoadCodebook(cfg.CodebookPath); err == nil {
						cand = AttachNearestRawBase(cand, cb)
					}
				}
				fits, _ := evaluateBatchParallel([]*Candidate{cand}, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
				scores = append(scores, algoScore{algo, fits[0], cand})
			}

			// Sort by distance, take top 4
			sort.Slice(scores, func(i, j int) bool { return scores[i].dist < scores[j].dist })
			for i := 0; i < 4 && i < len(scores); i++ {
				nnCandidates = append(nnCandidates, scores[i].cand.Copy())
			}

			// Early return check if NN prediction is exact (dist <= 0.0001)
			if scores[0].dist <= 1e-4 {
				if cfg.Verbose {
					fmt.Printf("Exact NN prediction found in Step 2 (dist=%.6f <= 0.0001). Returning early!\n", scores[0].dist)
				}
				reportProgress(cfg, 100, "Completed (Exact NN Match)")
				return Result{
					Best:     scores[0].cand,
					BestDist: scores[0].dist,
					Iters:    32,
					Top:      []*Candidate{scores[0].cand},
				}
			}

			// Quick live optimization & differential Jacobian steering on best NN result
			var globalLiveOpt *LiveOptimizer
			if cfg.Verbose {
				fmt.Println("--- NN Pre-optimization (live optimizer & diff optimizer) ---")
			}
			// Prefer codebook best if it beats the NN prediction
			initialPatch := scores[0].cand
			if cfg.Verbose {
				fmt.Printf("[debug] bestCodebookCand=%v bestCodebookDist=%.6f scores[0].dist=%.6f\n",
					bestCodebookCand != nil, bestCodebookDist, scores[0].dist)
			}
			if bestCodebookCand != nil && bestCodebookDist < scores[0].dist {
				initialPatch = bestCodebookCand
				if cfg.Verbose {
					fmt.Printf("Using codebook candidate as initial (dist=%.6f < NN dist=%.6f)\n",
						bestCodebookDist, scores[0].dist)
				}
			}
			globalLiveOpt = NewLiveOptimizer(target, initialPatch, 1000, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
			refined := globalLiveOpt.Run()

			// Differential Jacobian steering on extended temporal timbre features
			diffOpt := NewDiffOptimizer(target, refined, 5, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
			diffRefined := diffOpt.RunAlternating(2)
			if diffOpt.bestDist < globalLiveOpt.bestDist {
				refined = diffRefined
			}

			if globalLiveOpt.bestDist < scores[0].dist || diffOpt.bestDist < scores[0].dist {
				nnCandidates[0] = refined
				bestNnDist := globalLiveOpt.bestDist
				if diffOpt.bestDist < bestNnDist {
					bestNnDist = diffOpt.bestDist
				}
				if bestNnDist <= 1e-4 {
					if cfg.Verbose {
						fmt.Printf("Exact match found during NN pre-optimization (dist=%.6f <= 0.0001). Returning early!\n", bestNnDist)
					}
					reportProgress(cfg, 100, "Completed (Exact NN Match)")
					return Result{
						Best:     refined,
						BestDist: bestNnDist,
						Iters:    32 + 500,
						Top:      []*Candidate{refined},
					}
				}
			}
		}
	}

	// If no NN model was run, run pre-GA acceleration profiling on Waveform Base Seeds to seed GA islands
	var globalLiveOpt *LiveOptimizer
	if nnCandidates == nil {
		if cfg.Verbose {
			fmt.Println("--- Pre-GA Acceleration Profiling (Waveform Base Seeds) ---")
		}
		// Prefer codebook best as starting point if available
		startPatch := SineBase()
		if bestCodebookCand != nil {
			startPatch = bestCodebookCand
			if cfg.Verbose {
				fmt.Printf("Using codebook candidate as pre-GA initial (dist=%.6f)\n", bestCodebookDist)
			}
		}
		globalLiveOpt = NewLiveOptimizer(target, startPatch, 1000, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		globalLiveOpt.Run()
		if globalLiveOpt.bestDist <= 1e-4 {
			if cfg.Verbose {
				fmt.Printf("Exact match found during pre-GA profiling (dist=%.6f <= 0.0001). Returning early!\n", globalLiveOpt.bestDist)
			}
			reportProgress(cfg, 100, "Completed (Exact Pre-GA Match)")
			return Result{
				Best:     globalLiveOpt.bestPatch,
				BestDist: globalLiveOpt.bestDist,
				Iters:    500,
				Top:      []*Candidate{globalLiveOpt.bestPatch},
			}
		}
	}

	// Fallback: full GA pipeline
	popSize := cfg.GA.Population
	if popSize < 32 {
		popSize = 32
	}
	if popSize%32 != 0 {
		popSize = ((popSize / 32) + 1) * 32
	}
	cfg.GA.Population = popSize

	candidatesPerIsland := popSize / 32

	// Seed 32 Algorithm Islands
	islands := make([][]*Candidate, 32)
	islandFits := make([][]float64, 32)
	islandFeats := make([][]TargetFeatures, 32)
	islandGAs := make([]*GA, 32)

	totalEvals := 0

	if cfg.Verbose {
		fmt.Printf("Initializing 32 Algorithm Islands (%d candidates per island, total pop %d)...\n",
			candidatesPerIsland, popSize)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Initialize each island with candidates locked to its assigned algorithm (0..31)
	// Priority seeds (NN/KNN), acceleration trough permutations, and random/heuristic populace for exploration
	allInitialCandidates := make([]*Candidate, 0, popSize)
	for algo := 0; algo < 32; algo++ {
		islands[algo] = make([]*Candidate, candidatesPerIsland)
		for c := 0; c < candidatesPerIsland; c++ {
			var cand *Candidate

			// Priority 1: NN prediction for this algorithm's island (Slot 0)
			if nnCandidates != nil && c == 0 {
				for _, nn := range nnCandidates {
					if nn.Algorithm == uint8(algo) {
						cand = nn.Copy()
						break
					}
				}
			}

			// Priority 2: KNN codebook match (Slot 0)
			if cand == nil && c == 0 {
				if preset, ok := knnMap[uint8(algo)]; ok {
					cand = preset.Copy()
				}
			}

			// Priority 3: Acceleration Trough Permutation from LiveOptimizer (Slots 0..N/2)
			if cand == nil && globalLiveOpt != nil && c < (candidatesPerIsland+1)/2 {
				cand = globalLiveOpt.GeneratePermutationCandidate(rng, uint8(algo))
			}

			// Priority 4: Heuristic / Random candidate for exploration diversity (Slots N/2..N-1)
			if cand == nil {
				cand = HeuristicCandidateWithAlgo(features, cfg.NoteFreq, uint8(algo))
				if c > 0 {
					cand = cand.Copy()
					cand.Ops[c%6].FreqCoarse = uint8((int(cand.Ops[c%6].FreqCoarse) + c) % 32)
					cand.Ops[(c+1)%6].OutputLevel = uint8(rng.Intn(100))
				}
			}

			islands[algo][c] = cand
			allInitialCandidates = append(allInitialCandidates, cand)
		}
	}

	// Multi-core parallel evaluation of initial population across all CPU cores
	initialFits, initialFeats := evaluateBatchParallel(allInitialCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	totalEvals += len(allInitialCandidates)

	// Distribute initial fitnesses back to islands
	idx := 0
	for algo := 0; algo < 32; algo++ {
		islandFits[algo] = make([]float64, candidatesPerIsland)
		islandFeats[algo] = make([]TargetFeatures, candidatesPerIsland)
		for c := 0; c < candidatesPerIsland; c++ {
			islandFits[algo][c] = initialFits[idx]
			islandFeats[algo][c] = initialFeats[idx]
			idx++
		}
		islandGAConfig := cfg.GA
		islandGAConfig.Population = candidatesPerIsland
		islandGAs[algo] = NewGA(islands[algo], islandFits[algo], islandFeats[algo], islandGAConfig)
	}

	nicheGenerations := 15
	if cfg.GA.Generations <= 20 {
		nicheGenerations = cfg.GA.Generations * 3 / 4
		if nicheGenerations >= cfg.GA.Generations {
			nicheGenerations = cfg.GA.Generations - 1
		}
	} else if cfg.GA.Generations-nicheGenerations < 5 {
		nicheGenerations = cfg.GA.Generations - 5
	}
	if nicheGenerations < 1 {
		nicheGenerations = 1
	}

	// --- PHASE 1: Niche Island Evolution (Generations 0 to 20) ---
	if cfg.Verbose {
		fmt.Printf("--- Phase 1: Niche Island Evolution (%d generations across 32 algorithms) ---\n", nicheGenerations)
	}

	for gen := 0; gen < nicheGenerations; gen++ {
		p1 := 10 + (gen+1)*50/nicheGenerations
		reportProgress(cfg, p1, fmt.Sprintf("Phase 1: Niche Evolution (gen %d/%d)", gen+1, nicheGenerations))

		allGenCandidates := make([]*Candidate, 0, popSize)
		islandNextCandIdx := make([][]int, 32)

		for algo := 0; algo < 32; algo++ {
			nextCands := islandGAs[algo].Evolve(features)
			// Lock algorithm during niche phase
			for _, c := range nextCands {
				c.Algorithm = uint8(algo)
			}
			islandNextCandIdx[algo] = make([]int, len(nextCands))
			for cIdx, cand := range nextCands {
				islandNextCandIdx[algo][cIdx] = len(allGenCandidates)
				allGenCandidates = append(allGenCandidates, cand)
			}
		}

		genFits, genFeats := evaluateBatchParallel(allGenCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		totalEvals += len(allGenCandidates)

		// Update each island's GA state
		for algo := 0; algo < 32; algo++ {
			indices := islandNextCandIdx[algo]
			cands := make([]*Candidate, len(indices))
			fits := make([]float64, len(indices))
			feats := make([]TargetFeatures, len(indices))
			for i, globalIdx := range indices {
				cands[i] = allGenCandidates[globalIdx]
				fits[i] = genFits[globalIdx]
				feats[i] = genFeats[globalIdx]
			}
			islandGAs[algo].SetPopulation(cands, fits, feats)
		}

		if cfg.Verbose && (gen+1)%5 == 0 {
			bestIslandFit := islandGAs[0].BestFitness()
			bestAlgo := 0
			for algo := 1; algo < 32; algo++ {
				if f := islandGAs[algo].BestFitness(); f < bestIslandFit {
					bestIslandFit = f
					bestAlgo = algo
				}
			}
			fmt.Printf("[niche gen %d/%d] best=%.6f (Alg %d)  evals=%d\n",
				gen+1, nicheGenerations, bestIslandFit, bestAlgo+1, totalEvals)
		}

		// Early termination: check if any island found exact match
		if checkExactMatch(islandGAs) {
			if cfg.Verbose {
				fmt.Printf("[niche gen %d/%d] EXACT MATCH FOUND — skipping remaining generations\n",
					gen+1, nicheGenerations)
			}
			break
		}
	}

	// --- PHASE 2: Global Migration & Fine-Tuning ---
	if cfg.Verbose {
		fmt.Printf("--- Phase 2: Global Convergence (%d generations) ---\n", cfg.GA.Generations-nicheGenerations)
	}

	// Collect top candidates from all islands into global population pool
	globalSeed := make([]*Candidate, 0, popSize)
	globalFits := make([]float64, 0, popSize)
	globalFeats := make([]TargetFeatures, 0, popSize)

	for algo := 0; algo < 32; algo++ {
		best := islandGAs[algo].Best()
		bestFit := islandGAs[algo].BestFitness()
		globalSeed = append(globalSeed, best.Copy())
		globalFits = append(globalFits, bestFit)
		globalFeats = append(globalFeats, islandGAs[algo].currentFeatures[0])
	}

	// Run Operator Mini-GA to cross-breed modular operator blocks across island winners
	if len(globalSeed) > 0 {
		// Find the true global best across all 32 islands
		globalBestFit := globalFits[0]
		for _, f := range globalFits {
			if f < globalBestFit {
				globalBestFit = f
			}
		}

		miniGABest := RunOperatorMiniGA(globalSeed, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, 10, 32, cfg.Verbose)
		if miniGABest != nil && miniGABest.Fitness < globalBestFit {
			if cfg.Verbose {
				fmt.Printf("[Phase 1->2 Transition] Operator Mini-GA created hybrid candidate: dist=%.6f (improved from %.6f)\n",
					miniGABest.Fitness, globalBestFit)
			}
			// Find the slot with the worst fitness to replace (preserving diversity)
			worstIdx := 0
			for i, f := range globalFits {
				if f > globalFits[worstIdx] {
					worstIdx = i
				}
			}
			globalSeed[worstIdx] = miniGABest.Copy()
			globalFits[worstIdx] = miniGABest.Fitness
		}
	}

	for len(globalSeed) < popSize {
		for algo := 0; algo < 32 && len(globalSeed) < popSize; algo++ {
			top := islandGAs[algo].TopN(2)
			if len(top) > 1 {
				globalSeed = append(globalSeed, top[1].Copy())
				globalFits = append(globalFits, islandGAs[algo].currentFit[1])
				globalFeats = append(globalFeats, islandGAs[algo].currentFeatures[1])
			}
		}
	}

	globalGA := NewGA(globalSeed, globalFits, globalFeats, cfg.GA)

	// Stagnation tracking for codebook mutant injection
	stagnationTracker := NewStagnationTracker(6, 0.0001)
	rng = rand.New(rand.NewSource(42))
	var pendingMutants []*Candidate
	var boostGens int // generations to boost mutation rate after injection
	maxInject := 8

	for gen := nicheGenerations; gen < cfg.GA.Generations; gen++ {
		p2 := 60 + (gen-nicheGenerations+1)*25/(cfg.GA.Generations-nicheGenerations)
		reportProgress(cfg, p2, fmt.Sprintf("Phase 2: Global Convergence (gen %d/%d)", gen-nicheGenerations+1, cfg.GA.Generations-nicheGenerations))

		// Temporarily boost mutation rate after stagnation injection
		if boostGens > 0 {
			globalGA.config.MutationRateStart = 0.7
			boostGens--
		} else {
			globalGA.config.MutationRateStart = cfg.GA.MutationRateStart
		}

		nextCandidates := globalGA.Evolve(features)

		// Inject pending mutants from previous stagnation detection
		if len(pendingMutants) > 0 {
			replaceWeakestSorted(nextCandidates, globalGA.currentFit, pendingMutants, len(globalGA.eliteCandidates))
			if cfg.Verbose {
				fmt.Printf("[global gen %d/%d] injected %d codebook mutants\n",
					gen+1, cfg.GA.Generations, len(pendingMutants))
			}
			pendingMutants = nil
		}

		nextFitnesses, nextFeatures := evaluateBatchParallel(nextCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		totalEvals += len(nextCandidates)

		// Immediate 1D Line Search Exploitation
		for i := 0; i < len(nextCandidates); i++ {
			record := globalGA.mutations[i]
			if record.ParentIdx == -1 || record.ParamIdx == -1 {
				continue
			}
			parentFit := globalGA.currentFit[record.ParentIdx]
			if nextFitnesses[i] < parentFit*0.98 && nextFitnesses[i] < 99.0 {
				lineCands := make([]*Candidate, 2)
				for s := 0; s < 2; s++ {
					c := nextCandidates[i].Copy()
					magnitude := 2 + s
					step := record.Delta * int8(magnitude)
					applyStep(c, record.ParamIdx, step)
					lineCands[s] = c
				}
				lineFits, lineFeats := evaluateBatchParallel(lineCands, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
				totalEvals += len(lineCands)
				for s := 0; s < 2; s++ {
					if lineFits[s] < nextFitnesses[i] {
						nextCandidates[i] = lineCands[s]
						nextFitnesses[i] = lineFits[s]
						nextFeatures[i] = lineFeats[s]
					}
				}
			}
		}

		globalGA.SetPopulation(nextCandidates, nextFitnesses, nextFeatures)
		stagnationTracker.Record(globalGA.BestFitness())

		// Check stagnation and queue mutants for next generation
		if stagnationTracker.IsStagnant() && codebookPool != nil && codebookPool.Remaining() > 0 {
			n := maxInject
			if n > codebookPool.Remaining() {
				n = codebookPool.Remaining()
			}
			pendingMutants = codebookPool.Pop(n, rng)
			stagnationTracker.Reset()
			boostGens = 2 // boost mutation rate for 2 generations
			if cfg.Verbose {
				fmt.Printf("[global gen %d/%d] STAGNATION detected (best=%.6f plateaued for %d gens), queuing %d codebook mutants (%d remaining), mut boosted to 0.7 for 2 gens\n",
					gen+1, cfg.GA.Generations, globalGA.BestFitness(), stagnationTracker.PatienceThreshold, len(pendingMutants), codebookPool.Remaining())
			}
		}

		if cfg.Verbose && (gen+1)%10 == 0 {
			fmt.Printf("[global gen %d/%d] best=%.6f  mut=%.3f  evals=%d\n",
				gen+1, cfg.GA.Generations, globalGA.BestFitness(), globalGA.MutationRate(), totalEvals)
		}

		// Early termination: exact match found
		if globalGA.bestEverFit <= 1e-4 {
			if cfg.Verbose {
				fmt.Printf("[global gen %d/%d] EXACT MATCH FOUND (dist=%.6f <= 0.0001) — stopping early\n",
					gen+1, cfg.GA.Generations, globalGA.bestEverFit)
			}
			break
		}
	}

	// Final refinement: live optimizer & differential Jacobian steering with checkpoint backtracking
	reportProgress(cfg, 85, "Phase 3: Multi-Stage Hill-Climbing Optimization")
	if cfg.Verbose {
		fmt.Println("--- Phase 3: Live Optimizer & Differential Jacobian Steering ---")
	}

	best := globalGA.Best()
	gaBestDist := globalGA.BestFitness()
	liveOpt := NewLiveOptimizer(target, best, 500, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	finalPatch := liveOpt.Run()
	finalDist := liveOpt.bestDist

	// Differential Jacobian steering on extended temporal timbre features (alternating pass)
	diffOpt := NewDiffOptimizer(target, finalPatch, 5, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	diffPatch := diffOpt.RunAlternating(3)
	if diffOpt.bestDist < finalDist {
		finalPatch = diffPatch
		finalDist = diffOpt.bestDist
	}

	if cfg.Verbose {
		fmt.Printf("Phase 3 final distance: %.6f (GA was: %.6f, LiveOpt was: %.6f, DiffOpt was: %.6f)\n",
			finalDist, gaBestDist, liveOpt.bestDist, diffOpt.bestDist)
	}

	// Use refined result if it beat the GA
	if finalDist < gaBestDist {
		best = finalPatch
		best.Fitness = finalDist
	}

	topN := globalGA.TopN(32)
	// Ensure best is in topN and deduplicated
	bestSig := string(best.Encode())
	found := false
	for _, c := range topN {
		if string(c.Encode()) == bestSig {
			found = true
			break
		}
	}
	if !found {
		topN = append([]*Candidate{best}, topN...)
		if len(topN) > 32 {
			topN = topN[:32]
		}
	}

	// Calibrate carrier OutputLevels to match target RMS natively before export
	best.CalibrateCarrierLevels(rawTargetRMS, cfg.MIDINote, cfg.RenderDur)
	for _, c := range topN {
		c.CalibrateCarrierLevels(rawTargetRMS, cfg.MIDINote, cfg.RenderDur)
	}

	// -------------------------------------------------------------------------
	// Step 7: Timbral Top Selection — top 10 by static SpectralDistance, dedup to ≤5
	// Step 8: Full Top Selection — DiffOptimizer on those 10, dedup to ≤5 by temporal distance
	// -------------------------------------------------------------------------
	timbralTop5, fullTop5 := selectTopCandidates(topN, target, cfg, rawTargetRMS)

	reportProgress(cfg, 100, "Completed")

	return Result{
		Best:        best,
		BestDist:    best.Fitness,
		Iters:       totalEvals,
		Top:         topN,
		TimbralTop5: timbralTop5,
		FullTop5:    fullTop5,
	}
}

// selectTopCandidates implements Step 7 (timbral top selection) and Step 8 (full top selection).
// Step 7: renders topN candidates, sorts by static SpectralDistance, deduplicates to ≤5.
// Step 8: runs DiffOptimizer on up to 10 candidates, sorts by TemporalTimbreDistance, deduplicates to ≤5.
func selectTopCandidates(topN []*Candidate, target []float64, cfg Config, rawTargetRMS float64) (timbralTop5, fullTop5 []*Candidate) {
	const timbralLimit = 5
	const fullLimit = 5
	const expandPool = 10 // feed more than 5 into temporal expansion to guarantee ≥5 after dedup

	probeCount := len(topN)
	if probeCount > expandPool {
		probeCount = expandPool
	}
	if probeCount == 0 {
		return nil, nil
	}

	probes := topN[:probeCount]

	// --- Step 7: Timbral Top Selection ---
	type timbralEntry struct {
		cand *Candidate
		spec TemporalTimbreSpec
		dist float64
	}

	timbrePool := make([]timbralEntry, len(probes))
	for i, c := range probes {
		pcm := RenderCandidate(c, cfg.MIDINote, cfg.RenderDur)
		norm := NormalizeRMS(pcm)
		spec := ExtractTemporalTimbreSpec(norm, 44100)
		dist := SpectralDistance(target, norm)
		timbrePool[i] = timbralEntry{cand: c, spec: spec, dist: dist}
	}

	sort.Slice(timbrePool, func(i, j int) bool { return timbrePool[i].dist < timbrePool[j].dist })

	staticThreshold := cfg.StaticDedupThreshold
	if staticThreshold <= 0 {
		staticThreshold = 0.015
	}

	// Greedy deduplication on static specs
	var staticSelected []timbralEntry
	for _, item := range timbrePool {
		tooSimilar := false
		for _, sel := range staticSelected {
			if staticOnlyDist(sel.spec, item.spec) < staticThreshold {
				tooSimilar = true
				break
			}
		}
		if !tooSimilar {
			staticSelected = append(staticSelected, item)
			if len(staticSelected) >= timbralLimit {
				break
			}
		}
	}
	timbralTop5 = make([]*Candidate, len(staticSelected))
	for i, s := range staticSelected {
		timbralTop5[i] = s.cand
	}

	// --- Step 8: Full Top Selection (DiffOptimizer on the pool) ---
	type fullEntry struct {
		cand *Candidate
		spec TemporalTimbreSpec
		dist float64
	}

	targetSpec := ExtractTemporalTimbreSpec(target, 44100)
	fullPool := make([]fullEntry, 0, len(probes))

	for i, c := range probes {
		reportProgress(cfg, 85+i*1, fmt.Sprintf("Step 8: Temporal expansion %d/%d", i+1, len(probes)))

		diffOpt := NewDiffOptimizer(target, c, 2, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		diffOpt.SetSkipTopology(true)
		_ = diffOpt.RunAlternating(1) // 1 cycle: fast timbre + temporal pass

		optimized := diffOpt.BestPatch()
		optimized.Fitness = diffOpt.BestDist()
		spec := diffOpt.BestSpec() // cached — no re-render
		dist := TemporalTimbreDistance(targetSpec, spec)

		fullPool = append(fullPool, fullEntry{cand: optimized, spec: spec, dist: dist})
	}

	sort.Slice(fullPool, func(i, j int) bool { return fullPool[i].dist < fullPool[j].dist })

	temporalThreshold := cfg.TemporalDedupThreshold
	if temporalThreshold <= 0 {
		temporalThreshold = 0.020
	}

	// Greedy deduplication on temporal specs
	var temporalSelected []fullEntry
	for _, item := range fullPool {
		tooSimilar := false
		for _, sel := range temporalSelected {
			if TemporalTimbreDistance(sel.spec, item.spec) < temporalThreshold {
				tooSimilar = true
				break
			}
		}
		if !tooSimilar {
			temporalSelected = append(temporalSelected, item)
			if len(temporalSelected) >= fullLimit {
				break
			}
		}
	}
	fullTop5 = make([]*Candidate, len(temporalSelected))
	for i, s := range temporalSelected {
		fullTop5[i] = s.cand
	}

	return timbralTop5, fullTop5
}

// staticOnlyDist computes Euclidean distance between two TemporalTimbreSpec using only the 256-bin static component.
func staticOnlyDist(a, b TemporalTimbreSpec) float64 {
	var sum float64
	for i := 0; i < TimbreSpecBins; i++ {
		d := a.StaticSpec[i] - b.StaticSpec[i]
		sum += d * d
	}
	return math.Sqrt(sum) / math.Sqrt(float64(TimbreSpecBins))
}

// candidateScore tracks a candidate's distances from both codebooks for merged scoring.
type candidateScore struct {
	cand       *Candidate
	normFP     float64
	normTim    float64
	hasFP      bool
	hasTim     bool
	finalScore float64
}

// mergeCodebookResults merges fingerprint and timbre KNN results with preference scoring.
// Candidates found in BOTH codebooks get a bonus (lower score = better).
// Distances are min-max normalized per query batch to handle varying scales.
func mergeCodebookResults(
	fpCands []*Candidate, fpDists []float64,
	timCands []*Candidate, timDists []float64,
	topK int, bonus float64,
) []*Candidate {
	// Min-max normalize a distance slice to [0, 1]
	norm := func(dists []float64) []float64 {
		if len(dists) == 0 {
			return nil
		}
		min, max := dists[0], dists[0]
		for _, v := range dists {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		span := max - min
		if span < 1e-6 {
			span = 1.0
		}
		res := make([]float64, len(dists))
		for i, v := range dists {
			res[i] = (v - min) / span
		}
		return res
	}

	normFP := norm(fpDists)
	normTim := norm(timDists)

	scores := make(map[string]*candidateScore)

	// Index fingerprint results
	for i, c := range fpCands {
		sig := string(c.Encode())
		scores[sig] = &candidateScore{
			cand:   c,
			normFP: normFP[i],
			hasFP:  true,
		}
	}

	// Index timbre results, detect overlaps
	for i, c := range timCands {
		sig := string(c.Encode())
		if entry, exists := scores[sig]; exists {
			// Candidate appears in BOTH codebooks — dual-agreement
			entry.normTim = normTim[i]
			entry.hasTim = true
		} else {
			scores[sig] = &candidateScore{
				cand:    c,
				normTim: normTim[i],
				hasTim:  true,
			}
		}
	}

	// Calculate final scores
	mergedList := make([]*candidateScore, 0, len(scores))
	for _, entry := range scores {
		if entry.hasFP && entry.hasTim {
			// Dual-agreement: 50/50 blend with preference bonus
			entry.finalScore = 0.5*entry.normFP + 0.5*entry.normTim - bonus
		} else if entry.hasFP {
			entry.finalScore = entry.normFP
		} else {
			entry.finalScore = entry.normTim
		}
		mergedList = append(mergedList, entry)
	}

	// Sort ascending by finalScore (lower = better)
	sort.Slice(mergedList, func(i, j int) bool {
		return mergedList[i].finalScore < mergedList[j].finalScore
	})

	if topK > len(mergedList) {
		topK = len(mergedList)
	}
	result := make([]*Candidate, topK)
	for i := 0; i < topK; i++ {
		result[i] = mergedList[i].cand
	}

	return result
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

// runMicroGA runs a small GA fine-tuning pass on a predicted candidate.
// Only mutates OutputLevel, FreqFine, and Feedback. Algorithm is locked.
func runMicroGA(seed *Candidate, target []float64, features TargetFeatures, cfg Config) Result {
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	const microPop = 32
	const microGens = 5

	// Create population: 1 seed + 31 mutated copies
	rng := rand.New(rand.NewSource(42))
	pop := make([]*Candidate, microPop)
	pop[0] = seed.Copy()
	for i := 1; i < microPop; i++ {
		pop[i] = microMutate(seed.Copy(), rng)
	}

	// Evaluate initial population
	fits, feats := evaluateBatchParallel(pop, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	totalEvals := len(pop)

	// Track best
	bestDist := fits[0]
	bestIdx := 0
	for i := 1; i < microPop; i++ {
		if fits[i] < bestDist {
			bestDist = fits[i]
			bestIdx = i
		}
	}
	bestEverDist := bestDist
	bestEver := pop[bestIdx].Copy()

	// Run micro-GA
	microCfg := GAConfig{
		Population:        microPop,
		Generations:       microGens,
		TournamentK:       3,
		CrossoverRate:     0.5,
		MutationRateStart: 0.4,
		MutationRateEnd:   0.1,
		Elites:            2,
	}

	microGA := NewGA(pop, fits, feats, microCfg)

	for gen := 0; gen < microGens; gen++ {
		nextCandidates := microGA.Evolve(features)
		nextFitnesses, nextFeatures := evaluateBatchParallel(nextCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		totalEvals += len(nextCandidates)

		microGA.SetPopulation(nextCandidates, nextFitnesses, nextFeatures)

		// Track best
		for i := 0; i < microPop; i++ {
			if nextFitnesses[i] < bestEverDist {
				bestEverDist = nextFitnesses[i]
				bestEver = nextCandidates[i].Copy()
			}
		}

		if cfg.Verbose && (gen+1)%2 == 0 {
			fmt.Printf("  micro-gen %d/%d: best=%.6f\n", gen+1, microGens, bestEverDist)
		}

		// Early termination
		if bestEverDist < 0.01 {
			break
		}
	}

	return Result{
		Best:     bestEver,
		BestDist: bestEverDist,
		Iters:    totalEvals,
	}
}

// microMutate applies small perturbations to OutputLevel, FreqFine, and Feedback.
func microMutate(c *Candidate, rng *rand.Rand) *Candidate {
	d := c.Copy()

	for i := 0; i < 6; i++ {
		// OutputLevel ±4
		dl := int(rng.NormFloat64() * 2.0)
		nl := int(d.Ops[i].OutputLevel) + dl
		minLevel := 75
		if IsCarrier(d.Algorithm, i) {
			minLevel = 85
		}
		if nl < minLevel {
			nl = minLevel
		}
		if nl > 99 {
			nl = 99
		}
		d.Ops[i].OutputLevel = uint8(nl)

		// FreqFine ±2
		df := int(rng.NormFloat64() * 1.0)
		nf := int(d.Ops[i].FreqFine) + df
		if nf < 0 {
			nf = 0
		}
		if nf > 99 {
			nf = 99
		}
		d.Ops[i].FreqFine = uint8(nf)
	}

	// Feedback ±1
	if rng.Float64() < 0.30 {
		db := int(rng.NormFloat64() * 0.5)
		nb := int(d.Feedback) + db
		if nb < 0 {
			nb = 0
		}
		if nb > 7 {
			nb = 7
		}
		d.Feedback = uint8(nb)
	}

	return d
}

// runEnvelopeGA optimizes envelope parameters (rates, levels) while holding timbre fixed.
// Takes the best timbre candidate from Phase 1 and optimizes envelope shaping.
func runEnvelopeGA(timbreCand *Candidate, target []float64, features TargetFeatures, cfg Config) Result {
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	const envPop = 32
	const envGens = 10

	// Create population: clone the timbre candidate with envelope mutations
	rng := rand.New(rand.NewSource(43))
	pop := make([]*Candidate, envPop)
	pop[0] = timbreCand.Copy()

	for i := 1; i < envPop; i++ {
		pop[i] = envelopeMutate(timbreCand.Copy(), rng)
	}

	// Evaluate initial population
	fits, feats := evaluateBatchParallel(pop, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	totalEvals := len(pop)

	// Track best
	bestDist := fits[0]
	bestIdx := 0
	for i := 1; i < envPop; i++ {
		if fits[i] < bestDist {
			bestDist = fits[i]
			bestIdx = i
		}
	}
	bestEverDist := bestDist
	bestEver := pop[bestIdx].Copy()

	// Run envelope GA
	envCfg := GAConfig{
		Population:        envPop,
		Generations:       envGens,
		TournamentK:       3,
		CrossoverRate:     0.5,
		MutationRateStart: 0.3,
		MutationRateEnd:   0.1,
		Elites:            2,
	}

	envGA := NewGA(pop, fits, feats, envCfg)

	for gen := 0; gen < envGens; gen++ {
		nextCandidates := envGA.Evolve(features)
		nextFitnesses, nextFeatures := evaluateBatchParallel(nextCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
		totalEvals += len(nextCandidates)

		envGA.SetPopulation(nextCandidates, nextFitnesses, nextFeatures)

		for i := 0; i < envPop; i++ {
			if nextFitnesses[i] < bestEverDist {
				bestEverDist = nextFitnesses[i]
				bestEver = nextCandidates[i].Copy()
			}
		}

		if cfg.Verbose && (gen+1)%3 == 0 {
			fmt.Printf("  envelope-gen %d/%d: best=%.6f\n", gen+1, envGens, bestEverDist)
		}

		if bestEverDist < 0.01 {
			break
		}
	}

	return Result{
		Best:     bestEver,
		BestDist: bestEverDist,
		Iters:    totalEvals,
	}
}

// envelopeMutate perturbs only envelope params (rates, levels) while keeping timbre fixed.
func envelopeMutate(c *Candidate, rng *rand.Rand) *Candidate {
	d := c.Copy()

	for i := 0; i < 6; i++ {
		// Rate1 (attack) ±10
		dr1 := int(rng.NormFloat64() * 5.0)
		nr1 := int(d.Ops[i].Rate1) + dr1
		if nr1 < 0 {
			nr1 = 0
		}
		if nr1 > 99 {
			nr1 = 99
		}
		d.Ops[i].Rate1 = uint8(nr1)

		// Rate2 (decay) ±10
		dr2 := int(rng.NormFloat64() * 5.0)
		nr2 := int(d.Ops[i].Rate2) + dr2
		if nr2 < 0 {
			nr2 = 0
		}
		if nr2 > 99 {
			nr2 = 99
		}
		d.Ops[i].Rate2 = uint8(nr2)

		// Level3 (sustain) ±5
		dl3 := int(rng.NormFloat64() * 3.0)
		nl3 := int(d.Ops[i].Level3) + dl3
		if nl3 < 0 {
			nl3 = 0
		}
		if nl3 > 99 {
			nl3 = 99
		}
		d.Ops[i].Level3 = uint8(nl3)

		// Rate4 (release) ±10
		dr4 := int(rng.NormFloat64() * 5.0)
		nr4 := int(d.Ops[i].Rate4) + dr4
		if nr4 < 0 {
			nr4 = 0
		}
		if nr4 > 99 {
			nr4 = 99
		}
		d.Ops[i].Rate4 = uint8(nr4)
	}

	return d
}

// evaluateBatchParallel evaluates candidates concurrently across CPU cores using a multi-threaded worker pool.
func evaluateBatchParallel(candidates []*Candidate, target []float64, note uint8, dur time.Duration, rawTargetRMS float64) ([]float64, []TargetFeatures) {
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
				rawCandRMS := CalcRMS(trimmed)
				candidate := NormalizeRMS(trimmed)
				dist := SpectralDistance(target, candidate)
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
