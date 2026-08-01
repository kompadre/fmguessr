package match

import (
	"math"
	"math/rand"
)

// GAConfig holds genetic algorithm parameters.
type GAConfig struct {
	Population        int     // Population size (must be even, max 32)
	Generations       int     // Number of generations
	TournamentK       int     // Tournament selection size
	CrossoverRate     float64 // Probability of crossover
	MutationRateStart float64 // Initial mutation rate
	MutationRateEnd   float64 // Final mutation rate (decays over generations)
	Elites            int     // Number of elites preserved unchanged
}

// DefaultGAConfig returns sensible GA defaults.
func DefaultGAConfig() GAConfig {
	return GAConfig{
		Population:        32,
		Generations:       200,
		TournamentK:       3,
		CrossoverRate:     0.7,
		MutationRateStart: 0.7,
		MutationRateEnd:   0.05,
		Elites:            2,
	}
}

// MutationRecord tracks a mutation event for feedback.
type MutationRecord struct {
	ParentIdx int
	ParamIdx  int
	Delta     int8
}

// GA is a genetic algorithm optimizer for DX7 parameters.
type GA struct {
	rng             *rand.Rand
	config          GAConfig
	currentPop      []*Candidate
	currentFit      []float64
	currentFeatures []TargetFeatures
	eliteCandidates []*Candidate
	eliteFitness    []float64
	bestEver        *Candidate  // All-time best candidate (never lost)
	bestEverFit     float64     // All-time best fitness
	generation      int
	mutations       []MutationRecord // Records for the *next* evaluated population
}

// NewGA creates a genetic algorithm with a seed population.
func NewGA(seed []*Candidate, fitnesses []float64, features []TargetFeatures, cfg GAConfig) *GA {
	ga := &GA{
		rng:             rand.New(rand.NewSource(42)),
		config:          cfg,
		currentPop:      seed,
		currentFit:      fitnesses,
		currentFeatures: features,
		bestEverFit:     math.MaxFloat64,
	}
	ga.updateElites()

	// Initialize bestEver from seed population
	for i, f := range fitnesses {
		if f < ga.bestEverFit {
			ga.bestEverFit = f
			ga.bestEver = seed[i].Copy()
			ga.bestEver.Fitness = f
		}
	}

	return ga
}

// Best returns the best candidate found so far (all-time best).
func (ga *GA) Best() *Candidate {
	if ga.bestEver != nil {
		return ga.bestEver
	}
	bestIdx := 0
	for i := 1; i < len(ga.currentFit); i++ {
		if ga.currentFit[i] < ga.currentFit[bestIdx] {
			bestIdx = i
		}
	}
	return ga.currentPop[bestIdx]
}

// BestFitness returns the best fitness value (all-time best).
func (ga *GA) BestFitness() float64 {
	best := ga.currentFit[0]
	for _, f := range ga.currentFit {
		if f < best {
			best = f
		}
	}
	if ga.bestEverFit < best {
		best = ga.bestEverFit
	}
	return best
}

// Evolve produces the next generation of candidates.
func (ga *GA) Evolve(targetF TargetFeatures) []*Candidate {
	ga.generation++

	n := len(ga.currentPop)
	newPop := make([]*Candidate, n)
	ga.mutations = make([]MutationRecord, n)

	numElites := len(ga.eliteCandidates)

	// Elitism: include top elites unchanged
	for i := 0; i < numElites; i++ {
		newPop[i] = ga.eliteCandidates[i].Copy()
		ga.mutations[i] = MutationRecord{ParentIdx: -1} // No mutation feedback for elites
	}

	// Adaptive mutation rate
	t := float64(ga.generation) / float64(ga.config.Generations)
	mutationRate := ga.config.MutationRateStart + t*(ga.config.MutationRateEnd-ga.config.MutationRateStart)

	// Fill rest with offspring
	for i := numElites; i < n; i++ {
		idxA := ga.tournamentSelectIdx()
		idxB := ga.tournamentSelectIdx()
		parentA := ga.currentPop[idxA]
		parentB := ga.currentPop[idxB]

		var child *Candidate
		if ga.rng.Float64() < ga.config.CrossoverRate {
			child = Crossover(parentA, parentB, ga.rng)
		} else {
			child = parentA.Copy()
		}

		ga.mutations[i] = MutationRecord{ParentIdx: idxA, ParamIdx: -1}

		if ga.rng.Float64() < mutationRate {
			candF := ga.currentFeatures[idxA]
			bestFit := ga.BestFitness()
			mutant, pIdx, delta := child.MutateInformed(ga.rng, targetF, candF, bestFit)
			child = mutant
			ga.mutations[i].ParamIdx = pIdx
			ga.mutations[i].Delta = delta
		}

		newPop[i] = child
	}

	return newPop
}

// SetPopulation updates the population and applies momentum feedback.
func (ga *GA) SetPopulation(candidates []*Candidate, fitnesses []float64, features []TargetFeatures) {
	// Apply feedback based on mutation success
	for i, child := range candidates {
		if i >= len(ga.mutations) {
			continue
		}
		record := ga.mutations[i]
		if record.ParentIdx == -1 || record.ParamIdx == -1 {
			continue
		}

		parentFit := ga.currentFit[record.ParentIdx]
		childFit := fitnesses[i]
		p := record.ParamIdx

		if childFit < parentFit {
			// Improvement! Reinforce direction and boost velocity
			child.Direction[p] = 1
			if record.Delta < 0 {
				child.Direction[p] = -1
			}
			child.Velocity[p] *= 1.5
			if child.Velocity[p] > 8.0 {
				child.Velocity[p] = 8.0
			}
		} else {
			// Regression. Flip direction and decay velocity
			child.Direction[p] = -child.Direction[p]
			child.Velocity[p] *= 0.6
			if child.Velocity[p] < 0.2 {
				child.Velocity[p] = 0.2
			}
		}
	}

	ga.currentPop = candidates
	ga.currentFit = fitnesses
	ga.currentFeatures = features

	// Track all-time best
	for i, f := range fitnesses {
		if f < ga.bestEverFit {
			ga.bestEverFit = f
			ga.bestEver = candidates[i].Copy()
			ga.bestEver.Fitness = f
		}
	}

	// Elite rescue: if bestEver is not in the current population, inject it
	if ga.bestEver != nil {
		found := false
		for _, f := range fitnesses {
			if f <= ga.bestEverFit+1e-10 {
				found = true
				break
			}
		}
		if !found {
			// Find worst candidate and replace with bestEver
			worstIdx := 0
			for i := 1; i < len(fitnesses); i++ {
				if fitnesses[i] > fitnesses[worstIdx] {
					worstIdx = i
				}
			}
			candidates[worstIdx] = ga.bestEver.Copy()
			fitnesses[worstIdx] = ga.bestEverFit
			ga.currentPop = candidates
			ga.currentFit = fitnesses
		}
	}

	ga.updateElites()
}

// MutationRate returns the current adaptive mutation rate.
func (ga *GA) MutationRate() float64 {
	t := float64(ga.generation) / float64(ga.config.Generations)
	return ga.config.MutationRateStart + t*(ga.config.MutationRateEnd-ga.config.MutationRateStart)
}

// updateElites selects the best individuals from the current evaluated population.
func (ga *GA) updateElites() {
	n := len(ga.currentPop)
	if n == 0 {
		return
	}

	// Find indices sorted by fitness
	type idxFit struct {
		idx int
		fit float64
	}
	sorted := make([]idxFit, n)
	for i := range ga.currentPop {
		sorted[i] = idxFit{i, ga.currentFit[i]}
	}
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].fit < sorted[i].fit {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	numElites := ga.config.Elites
	if numElites > n {
		numElites = n
	}
	ga.eliteCandidates = make([]*Candidate, numElites)
	ga.eliteFitness = make([]float64, numElites)
	for i := 0; i < numElites; i++ {
		ga.eliteCandidates[i] = ga.currentPop[sorted[i].idx].Copy()
		ga.eliteFitness[i] = sorted[i].fit
	}
}

// TopN returns the N best unique candidates from the current population.
func (ga *GA) TopN(n int) []*Candidate {
	type idxFit struct {
		idx int
		fit float64
	}
	sorted := make([]idxFit, len(ga.currentPop))
	for i := range ga.currentPop {
		sorted[i] = idxFit{i, ga.currentFit[i]}
	}
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].fit < sorted[i].fit {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	seen := make(map[string]bool)
	var result []*Candidate
	for _, item := range sorted {
		cand := ga.currentPop[item.idx]
		sig := string(cand.Encode())
		if !seen[sig] {
			seen[sig] = true
			result = append(result, cand)
			if len(result) == n {
				break
			}
		}
	}
	return result
}

func (ga *GA) tournamentSelectIdx() int {
	bestIdx := -1
	var bestFit float64
	for i := 0; i < ga.config.TournamentK; i++ {
		idx := ga.rng.Intn(len(ga.currentPop))
		fit := ga.currentFit[idx]
		if bestIdx == -1 || fit < bestFit {
			bestIdx = idx
			bestFit = fit
		}
	}
	return bestIdx
}

func (ga *GA) tournamentSelect() *Candidate {
	return ga.currentPop[ga.tournamentSelectIdx()]
}
