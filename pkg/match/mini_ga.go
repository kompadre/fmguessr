package match

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

// RunOperatorMiniGA executes an ultra-fast local mini-GA (5-10 generations) that cross-breeds
// modular operator blocks across top candidate patches and algorithm topologies.
func RunOperatorMiniGA(ctx context.Context, parents []*Candidate, targetNorm []float64, note uint8, dur time.Duration, rawTargetRMS float64, generations int, popSize int, verbose bool) *Candidate {
	if len(parents) == 0 {
		return nil
	}
	if popSize < 8 {
		popSize = 16
	}
	if popSize%2 != 0 {
		popSize++
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Best overall tracking
	var bestCand *Candidate
	bestFit := math.MaxFloat64

	// Initialize mini-GA population
	pop := make([]*Candidate, popSize)

	// Seed first slots with parent copies
	for i := 0; i < len(parents) && i < popSize; i++ {
		pop[i] = parents[i].Copy()
	}

	// Fill remaining slots with operator block crossbreeds
	for i := len(parents); i < popSize; i++ {
		p1 := parents[rng.Intn(len(parents))]
		p2 := parents[rng.Intn(len(parents))]
		child := p1.Copy()
		child.Algorithm = uint8(rng.Intn(32))
		if rng.Intn(2) == 0 {
			child.Feedback = p2.Feedback
		}

		// Swap operator blocks
		for op := 0; op < 6; op++ {
			if rng.Intn(2) == 0 {
				child.Ops[op] = p2.Ops[op]
			}
		}
		pop[i] = child
	}

	// Evaluate initial mini-GA population
	fits, _ := evaluateBatchParallel(pop, targetNorm, note, dur, rawTargetRMS)
	for i, f := range fits {
		pop[i].Fitness = f
		if f < bestFit {
			bestFit = f
			bestCand = pop[i].Copy()
		}
	}

	if verbose {
		fmt.Printf("[mini-GA] Starting operator mini-GA across %d parents (initial best dist: %.6f)\n", len(parents), bestFit)
	}

	// Evolve mini-GA over generations
	for gen := 0; gen < generations; gen++ {
		// Check context cancellation (budget)
		select {
		case <-ctx.Done():
			if verbose {
				fmt.Printf("[mini-GA] CANCELLED at gen %d — stopping early\n", gen+1)
			}
			return bestCand
		default:
		}

		// Sort population by fitness
		type popFit struct {
			cand *Candidate
			fit  float64
		}
		sorted := make([]popFit, popSize)
		for i := 0; i < popSize; i++ {
			sorted[i] = popFit{cand: pop[i], fit: fits[i]}
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].fit < sorted[j].fit })

		// Elitism: keep top 2 candidates
		nextPop := make([]*Candidate, popSize)
		nextPop[0] = sorted[0].cand.Copy()
		nextPop[1] = sorted[1].cand.Copy()

		// Fill remaining with operator block crossover offspring
		for i := 2; i < popSize; i++ {
			pA := sorted[rng.Intn(min(8, popSize))].cand
			pB := sorted[rng.Intn(min(8, popSize))].cand

			child := pA.Copy()

			// Algorithm inheritance or mutation
			if rng.Float64() < 0.3 {
				child.Algorithm = uint8(rng.Intn(32))
			} else if rng.Intn(2) == 0 {
				child.Algorithm = pB.Algorithm
			}

			if rng.Intn(2) == 0 {
				child.Feedback = pB.Feedback
			}

			// Crossbreed operator blocks
			for op := 0; op < 6; op++ {
				if rng.Intn(2) == 0 {
					child.Ops[op] = pB.Ops[op]
				}
			}

			// Light mutation on 20% of offspring
			if rng.Float64() < 0.20 {
				mOp := rng.Intn(6)
				switch rng.Intn(3) {
				case 0:
					// Adjust output level
					newLvl := int(child.Ops[mOp].OutputLevel) + (rng.Intn(7) - 3)
					if newLvl < 0 {
						newLvl = 0
					}
					if newLvl > 99 {
						newLvl = 99
					}
					child.Ops[mOp].OutputLevel = uint8(newLvl)
				case 1:
					// Adjust coarse frequency
					newC := int(child.Ops[mOp].FreqCoarse) + (rng.Intn(3) - 1)
					if newC < 0 {
						newC = 0
					}
					if newC > 31 {
						newC = 31
					}
					child.Ops[mOp].FreqCoarse = uint8(newC)
				case 2:
					// Adjust fine frequency
					newF := int(child.Ops[mOp].FreqFine) + (rng.Intn(9) - 4)
					if newF < 0 {
						newF = 0
					}
					if newF > 99 {
						newF = 99
					}
					child.Ops[mOp].FreqFine = uint8(newF)
				}
			}

			nextPop[i] = child
		}

		// Parallel batch evaluation of generation offspring
		pop = nextPop
		fits, _ = evaluateBatchParallel(pop, targetNorm, note, dur, rawTargetRMS)

		for i, f := range fits {
			pop[i].Fitness = f
			if f < bestFit {
				bestFit = f
				bestCand = pop[i].Copy()
				if verbose {
					fmt.Printf("[mini-GA] gen %d: new best dist %.6f (Alg %d)\n", gen+1, bestFit, bestCand.Algorithm+1)
				}
			}
		}
	}

	return bestCand
}
