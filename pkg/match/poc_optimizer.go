package match

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// POCOptimizer is a standalone proof-of-concept optimizer runner designed to evaluate
// multi-knob trajectory vectoring, coupled operator sweeps, and multi-dial cross-trough sampling
// in isolation without modifying live_optimizer.go.
type POCOptimizer struct {
	targetNorm   []float64
	rawTargetRMS float64
	currentPatch *Candidate
	bestPatch    *Candidate
	bestDist     float64
	note         uint8
	dur          time.Duration
	maxIter      int
	verbose      bool
	accelCache   map[string][]int
	rng          *rand.Rand
}

func NewPOCOptimizer(target []float64, initial *Candidate, maxIter int, verbose bool, note uint8, dur time.Duration, rawTargetRMS ...float64) *POCOptimizer {
	targetNorm := NormalizeRMS(target)
	tRMS := 0.0
	if len(rawTargetRMS) > 0 && rawTargetRMS[0] > 0 {
		tRMS = rawTargetRMS[0]
	} else {
		tRMS = CalcRMS(target)
	}

	return &POCOptimizer{
		targetNorm:   targetNorm,
		rawTargetRMS: tRMS,
		currentPatch: initial.Copy(),
		bestPatch:    initial.Copy(),
		bestDist:     math.MaxFloat64,
		note:         note,
		dur:          dur,
		maxIter:      maxIter,
		verbose:      verbose,
		accelCache:   make(map[string][]int),
		rng:          rand.New(rand.NewSource(42)),
	}
}

func (poc *POCOptimizer) BestDist() float64 {
	return poc.bestDist
}

func (poc *POCOptimizer) BestPatch() *Candidate {
	return poc.bestPatch
}

func (poc *POCOptimizer) Run() *Candidate {
	knobs := buildKnobs(true)
	knobMap := make(map[string]KnobDef, len(knobs))
	for _, k := range knobs {
		knobMap[k.name] = k
	}

	// Initial evaluation
	initialFits, _ := evaluateBatchParallel([]*Candidate{poc.currentPatch}, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)
	if len(initialFits) > 0 {
		poc.bestDist = initialFits[0]
	}

	if poc.verbose {
		fmt.Printf("[POC-Start] Initial Distance: %.6f\n", poc.bestDist)
	}

	for iter := 0; iter < poc.maxIter; iter++ {
		improved := false

		// 1. Single-knob Coarse Sweeps & Improver Harvesting
		coarseCands, coarseMeta := generateBatch(poc.currentPatch, knobs, false)
		if len(coarseCands) > 0 {
			fits, _ := evaluateBatchParallel(coarseCands, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)

			// Update acceleration cache with coarse sweep second derivatives
			updateAccelerationCache(poc.accelCache, knobs, poc.bestDist, coarseCands, coarseMeta, fits)

			var improvers []knobDelta
			for i, fit := range fits {
				if fit < poc.bestDist {
					k, ok := knobMap[coarseMeta[i].knobName]
					if ok {
						curVal := k.getter(poc.currentPatch)
						delta := coarseMeta[i].val - curVal
						imp := poc.bestDist - fit
						improvers = append(improvers, knobDelta{
							knob:  k,
							delta: delta,
							imp:   imp,
						})
					}

					poc.bestDist = fit
					poc.bestPatch = coarseCands[i].Copy()
					poc.currentPatch = coarseCands[i].Copy()
					improved = true
					if poc.verbose {
						fmt.Printf("[POC-Coarse] Iter %d: New Best Dist %.6f via %s\n", iter+1, poc.bestDist, coarseMeta[i].knobName)
					}
				}
			}

			// 2. Multi-Knob Trajectory Candidates
			if len(improvers) >= 2 {
				trajCands, trajMeta := generateMultiKnobTrajectoryCandidates(poc.currentPatch, knobs, improvers)
				if len(trajCands) > 0 {
					trajFits, _ := evaluateBatchParallel(trajCands, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)
					for i, fit := range trajFits {
						if fit < poc.bestDist {
							poc.bestDist = fit
							poc.bestPatch = trajCands[i].Copy()
							poc.currentPatch = trajCands[i].Copy()
							improved = true
							if poc.verbose {
								fmt.Printf("[POC-Trajectory] Iter %d: New Best Dist %.6f via %s\n", iter+1, poc.bestDist, trajMeta[i].knobName)
							}
						}
					}
				}
			}
		}

		// 3. Coupled Operator Candidates
		coupledCands, coupledMeta := generateCoupledOperatorCandidates(poc.currentPatch, knobs)
		if len(coupledCands) > 0 {
			fits, _ := evaluateBatchParallel(coupledCands, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)
			for i, fit := range fits {
				if fit < poc.bestDist {
					poc.bestDist = fit
					poc.bestPatch = coupledCands[i].Copy()
					poc.currentPatch = coupledCands[i].Copy()
					improved = true
					if poc.verbose {
						fmt.Printf("[POC-Coupled] Iter %d: New Best Dist %.6f via %s\n", iter+1, poc.bestDist, coupledMeta[i].knobName)
					}
				}
			}
		}

		// 4. Multi-Dial Cross-Trough Candidates from Acceleration Cache
		crossCands, crossMeta := generateMultiDialCrossTroughCandidates(poc.currentPatch, knobs, poc.accelCache, poc.rng)
		if len(crossCands) > 0 {
			fits, _ := evaluateBatchParallel(crossCands, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)
			for i, fit := range fits {
				if fit < poc.bestDist {
					poc.bestDist = fit
					poc.bestPatch = crossCands[i].Copy()
					poc.currentPatch = crossCands[i].Copy()
					improved = true
					if poc.verbose {
						fmt.Printf("[POC-CrossTrough] Iter %d: New Best Dist %.6f via %s\n", iter+1, poc.bestDist, crossMeta[i].knobName)
					}
				}
			}
		}

		// 5. Fine Sweeps
		fineCands, fineMeta := generateBatch(poc.currentPatch, knobs, true)
		if len(fineCands) > 0 {
			fits, _ := evaluateBatchParallel(fineCands, poc.targetNorm, poc.note, poc.dur, poc.rawTargetRMS, nil)
			for i, fit := range fits {
				if fit < poc.bestDist {
					poc.bestDist = fit
					poc.bestPatch = fineCands[i].Copy()
					poc.currentPatch = fineCands[i].Copy()
					improved = true
					if poc.verbose {
						fmt.Printf("[POC-Fine] Iter %d: New Best Dist %.6f via %s\n", iter+1, poc.bestDist, fineMeta[i].knobName)
					}
				}
			}
		}

		if !improved {
			if poc.verbose {
				fmt.Printf("[POC-Stagnation] Iter %d: No further improvement, stopping.\n", iter+1)
			}
			break
		}
	}

	return poc.bestPatch
}
