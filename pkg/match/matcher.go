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
	RenderDur        time.Duration
	MIDINote         uint8
	NoteFreq         float64 // User-specified note frequency (0 = use detected F0)
	CodebookPath     string  // Path to pre-indexed dx7_codebook.bin (optional)
	ModelPath        string  // Path to ONNX model for NN prediction (optional)
	Verbose          bool
	GA               GAConfig
	ProgressCallback func(percent int, phase string) // Optional callback for API server progress reporting
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		RenderDur: 2 * time.Second,
		MIDINote:  60,
		Verbose:   false,
		GA:        DefaultGAConfig(),
	}
}

// Result holds the output of the optimization.
type Result struct {
	Best     *Candidate
	BestDist float64
	Iters    int
	Top      []*Candidate // Top 32 candidates for .syx export
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
		if cb, err := LoadCodebook(cfg.CodebookPath); err == nil && len(cb.Entries) > 0 {
			targetFp := ComputeFingerprint(target, RenderSampleRate)
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
					fmt.Printf("Step 1 Codebook Lookup: evaluated %d candidates (best dist=%.6f)\n",
						len(knnMatches), bestCodebookDist)
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
						Iters:    len(knnMatches),
						Top:      []*Candidate{bestCodebookCand},
					}
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
	if cfg.ModelPath != "" {
		predictor, err := NewPredictor(cfg.ModelPath)
		if err != nil {
			fmt.Printf("Warning: could not load NN model: %v\n", err)
		} else {
			defer predictor.Close()

			if cfg.Verbose {
				fmt.Printf("Using NN predictor: %s\n", cfg.ModelPath)
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
			globalLiveOpt = NewLiveOptimizer(target, scores[0].cand, 1000, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
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
		globalLiveOpt = NewLiveOptimizer(target, SineBase(), 1000, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
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
		miniGABest := RunOperatorMiniGA(globalSeed, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, 10, 32, cfg.Verbose)
		if miniGABest != nil && miniGABest.Fitness < globalFits[0] {
			if cfg.Verbose {
				fmt.Printf("[Phase 1->2 Transition] Operator Mini-GA created hybrid candidate: dist=%.6f (improved from %.6f)\n",
					miniGABest.Fitness, globalFits[0])
			}
			globalSeed[0] = miniGABest.Copy()
			globalFits[0] = miniGABest.Fitness
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

	reportProgress(cfg, 100, "Completed")

	return Result{
		Best:     best,
		BestDist: best.Fitness,
		Iters:    totalEvals,
		Top:      topN,
	}
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
