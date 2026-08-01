package match

import (
	"math"
	"math/rand"
	"sort"
)

// StagnationTracker detects when the GA has stalled and signals for codebook mutant injection.
type StagnationTracker struct {
	PatienceThreshold int     // Consecutive stagnant gens before triggering injection
	MinImprovement    float64 // Minimum fitness drop to reset the counter
	bestHistory        []float64
	counter            int
}

// NewStagnationTracker creates a tracker with the given patience and improvement thresholds.
func NewStagnationTracker(patience int, minImprovement float64) *StagnationTracker {
	return &StagnationTracker{
		PatienceThreshold: patience,
		MinImprovement:    minImprovement,
	}
}

// Record appends the current best fitness and updates the stagnation counter.
func (t *StagnationTracker) Record(bestFitness float64) {
	if len(t.bestHistory) > 0 {
		improvement := t.bestHistory[len(t.bestHistory)-1] - bestFitness
		if improvement < t.MinImprovement {
			t.counter++
		} else {
			t.counter = 0
		}
	}
	t.bestHistory = append(t.bestHistory, bestFitness)
}

// IsStagnant returns true if the GA has stalled for PatienceThreshold generations.
func (t *StagnationTracker) IsStagnant() bool {
	return t.counter >= t.PatienceThreshold
}

// Reset clears the stagnation counter after mutant injection.
func (t *StagnationTracker) Reset() {
	t.counter = 0
}

// CodebookPool manages unused codebook entries for stagnation injection.
// Entries are consumed without replacement so each mutant is unique.
type CodebookPool struct {
	entries    []*Candidate
	consumed   map[int]bool
	nextIdx    int
	allEntries []*Candidate // full codebook for refill
}

// NewCodebookPool creates a pool from unused codebook entries.
func NewCodebookPool(unusedEntries []*Candidate) *CodebookPool {
	cp := &CodebookPool{
		entries:    make([]*Candidate, len(unusedEntries)),
		consumed:   make(map[int]bool),
		allEntries: make([]*Candidate, len(unusedEntries)),
	}
	copy(cp.entries, unusedEntries)
	copy(cp.allEntries, unusedEntries)
	return cp
}

// Pop returns up to n unique mutated codebook entries.
// Consumed entries are not returned again until the pool is refilled.
func (cp *CodebookPool) Pop(n int, rng *rand.Rand) []*Candidate {
	if n <= 0 {
		return nil
	}

	// Collect unconsumed indices
	available := make([]int, 0)
	for i := range cp.entries {
		if !cp.consumed[i] {
			available = append(available, i)
		}
	}

	// Refill if exhausted
	if len(available) == 0 {
		cp.consumed = make(map[int]bool)
		available = make([]int, len(cp.entries))
		for i := range available {
			available[i] = i
		}
	}

	// Fisher-Yates partial shuffle to pick n random indices
	if n > len(available) {
		n = len(available)
	}
	for i := 0; i < n; i++ {
		j := i + rng.Intn(len(available)-i)
		available[i], available[j] = available[j], available[i]
	}

	mutants := make([]*Candidate, n)
	for i := 0; i < n; i++ {
		idx := available[i]
		cp.consumed[idx] = true
		mutants[i] = MutateCodebookPatch(cp.entries[idx], rng)
	}
	return mutants
}

// Remaining returns the number of unconsumed entries.
func (cp *CodebookPool) Remaining() int {
	count := 0
	for i := range cp.entries {
		if !cp.consumed[i] {
			count++
		}
	}
	return count
}

// MutateCodebookPatch creates a lightly mutated copy of a codebook candidate.
// Uses Gaussian perturbation on OutputLevel (σ=6), FreqFine (σ=2),
// and Feedback (σ=1.5, 30% chance). Fitness is set to MaxFloat64 (unevaluated).
func MutateCodebookPatch(cand *Candidate, rng *rand.Rand) *Candidate {
	mutant := cand.Copy()
	mutant.Fitness = math.MaxFloat64

	for opIdx := 0; opIdx < 6; opIdx++ {
		// Perturb OutputLevel (+/- 12 range)
		levelDelta := int(rng.NormFloat64() * 6.0)
		mutant.Ops[opIdx].OutputLevel = uint8(clampInt(
			int(mutant.Ops[opIdx].OutputLevel)+levelDelta, 0, 99))

		// Perturb FreqFine (+/- 4 range)
		fineDelta := int(rng.NormFloat64() * 2.0)
		mutant.Ops[opIdx].FreqFine = uint8(clampInt(
			int(mutant.Ops[opIdx].FreqFine)+fineDelta, 0, 99))
	}

	// Small chance to perturb feedback
	if rng.Float64() < 0.30 {
		fbDelta := int(rng.NormFloat64() * 1.5)
		mutant.Feedback = uint8(clampInt(
			int(mutant.Feedback)+fbDelta, 0, 7))
	}

	return mutant
}

// replaceWeakestSorted sorts candidates by fitness (worst first) and replaces
// the last len(mutants) entries with the given mutants.
// Elite candidates at the front of the slice (from Evolve) are preserved.
// Note: fitnesses is copied internally to avoid corrupting the caller's slice.
func replaceWeakestSorted(candidates []*Candidate, fitnesses []float64, mutants []*Candidate, numElites int) {
	n := len(candidates)
	m := len(mutants)
	if m == 0 {
		return
	}

	// Work on a copy so we never mutate the caller's fitness slice
	fitCopy := make([]float64, len(fitnesses))
	copy(fitCopy, fitnesses)

	// Build index list for the non-elite portion (indices numElites..n-1)
	nonElite := make([]int, n-numElites)
	for i := range nonElite {
		nonElite[i] = numElites + i
	}

	// Sort non-elite indices by fitness (worst first = highest fitness)
	sort.Slice(nonElite, func(i, j int) bool {
		return fitCopy[nonElite[i]] > fitCopy[nonElite[j]]
	})

	// Replace the m weakest non-elite candidates
	for i := 0; i < m && i < len(nonElite); i++ {
		idx := nonElite[i]
		candidates[idx] = mutants[i]
		fitnesses[idx] = math.MaxFloat64 // Mark as unevaluated
	}
}

func clampInt(val, min, max int) int {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}
