package match

import (
	"context"
	"math"
	"math/rand"
	"time"
)

type GAStep struct{}

func (s *GAStep) Name() string { return "GAStep" }

func (s *GAStep) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	popSize := cfg.GA.Population
	if popSize < 32 {
		popSize = 32
	}
	if popSize%32 != 0 {
		popSize = ((popSize / 32) + 1) * 32
	}
	cfg.GA.Population = popSize
	candidatesPerIsland := popSize / 32

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Use codebook knnMap from state (populated by CodebookSearch)
	knnMap := in.State.KNNMap
	if knnMap == nil {
		knnMap = make(map[uint8]*Candidate)
	}

	islands := make([][]*Candidate, 32)
	islandFits := make([][]float64, 32)
	islandFeats := make([][]TargetFeatures, 32)
	islandGAs := make([]*GA, 32)

	allInitialCandidates := make([]*Candidate, 0, popSize)
	for algo := 0; algo < 32; algo++ {
		islands[algo] = make([]*Candidate, candidatesPerIsland)
		for c := 0; c < candidatesPerIsland; c++ {
			var cand *Candidate

			// Priority 1: NN prediction for this algorithm's island (Slot 0)
			if len(in.State.NNCandidates) > 0 && c == 0 {
				for _, nn := range in.State.NNCandidates {
					if nn.Algorithm == uint8(algo) {
						cand = nn.Copy()
						break
					}
				}
			}

			// Priority 1.5: Beam search result for this algorithm (Slot 0)
			if cand == nil && c == 0 && in.State.BeamResult != nil && in.State.BeamResult.Final.Algorithm == uint8(algo) {
				cand = in.State.BeamResult.Final.Copy()
			}

			// Priority 2: KNN codebook match (Slot 0)
			if cand == nil && c == 0 {
				if preset, ok := knnMap[uint8(algo)]; ok {
					cand = preset.Copy()
				}
			}

			// Priority 3: LiveOptimizer permutation candidates (Slots 0..N/2)
			if cand == nil && in.State.LiveOpt != nil && c < (candidatesPerIsland+1)/2 {
				cand = in.State.LiveOpt.GeneratePermutationCandidate(rng, uint8(algo))
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

	initialFits, initialFeats := evaluateBatchParallel(ctx, allInitialCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
	totalEvals := len(allInitialCandidates)

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

	logElapsed(cfg, "--- Phase 1: Niche Island Evolution (%d generations across 32 algorithms) ---\n", nicheGenerations)
	gaStartTime := time.Now()

nicheLoop:
	for gen := 0; gen < nicheGenerations; gen++ {
		if ctx.Err() != nil {
			break nicheLoop
		}
		allGenCandidates := make([]*Candidate, 0, popSize)
		islandNextCandIdx := make([][]int, 32)
		for algo := 0; algo < 32; algo++ {
			nextCands := islandGAs[algo].Evolve(features)
			for _, c := range nextCands {
				c.Algorithm = uint8(algo)
			}
			islandNextCandIdx[algo] = make([]int, len(nextCands))
			for cIdx, cand := range nextCands {
				islandNextCandIdx[algo][cIdx] = len(allGenCandidates)
				allGenCandidates = append(allGenCandidates, cand)
			}
		}
		genFits, genFeats := evaluateBatchParallel(ctx, allGenCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
		totalEvals += len(allGenCandidates)
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
		if checkExactMatch(islandGAs) {
			break nicheLoop
		}
	}

	if ctx.Err() != nil {
		// Budget exhausted after niche phase — return best from niche islands
		bestIslandFit := math.MaxFloat64
		var bestIslandCand *Candidate
		for algo := 0; algo < 32; algo++ {
			if islandGAs[algo] != nil {
				ib := islandGAs[algo].Best()
				if ib != nil && ib.Fitness < bestIslandFit {
					bestIslandFit = ib.Fitness
					bestIslandCand = ib
				}
			}
		}
		if bestIslandCand != nil {
			result.Cohort.Best = *bestIslandCand
			result.Cohort.Best.Fitness = bestIslandFit
		}
		goto finalizeGA
	}

	{
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
		prevBestFit := math.MaxFloat64
		convergeGens := 0
		convergeGens_threshold := 0.001
		convergePatience := 5
		convergeMinGens := 20
		settleActive := false
		settleGens := 0
		settlePatience := 5

		logElapsed(cfg, "--- Phase 2: Global Convergence (%d generations) ---\n", cfg.GA.Generations-nicheGenerations)

		for gen := nicheGenerations; gen < cfg.GA.Generations; gen++ {
			if ctx.Err() != nil {
				break
			}
			nextCandidates := globalGA.Evolve(features)
			nextFitnesses, nextFeatures := evaluateBatchParallel(ctx, nextCandidates, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
			totalEvals += len(nextCandidates)
			globalGA.SetPopulation(nextCandidates, nextFitnesses, nextFeatures)

			if globalGA.bestEverFit <= 1e-4 {
				break
			}
			bestFit := globalGA.BestFitness()
			improvement := prevBestFit - bestFit
			if improvement < convergeGens_threshold {
				convergeGens++
			} else {
				convergeGens = 0
			}
			prevBestFit = bestFit
			if convergeGens >= convergePatience && gen >= convergeMinGens {
				if !settleActive {
					settleActive = true
					settleGens = 0
					convergeGens = 0
					continue
				}
				break
			}
			if settleActive {
				settleGens++
				if settleGens >= settlePatience {
					break
				}
			}
		}

		gaBest := globalGA.Best()
		gaBestDist := globalGA.BestFitness()
		result.Cohort.Best = *gaBest
		result.Cohort.Best.Fitness = gaBestDist

		topGA := globalGA.TopN(MaxCohortLen)
		for i := 0; i < len(topGA) && i < MaxCohortLen; i++ {
			result.Cohort.Timbral[i] = *topGA[i]
		}
		if len(topGA) < MaxCohortLen {
			result.Cohort.TimbralN = len(topGA)
		} else {
			result.Cohort.TimbralN = MaxCohortLen
		}
	}

finalizeGA:
	_ = gaStartTime
	return result
}
