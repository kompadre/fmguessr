package match

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

type LiveOptimizer struct {
	currentPatch *Candidate
	targetNorm   []float64
	note         uint8
	dur          time.Duration
	bestDist     float64
	bestPatch    *Candidate
	maxIter      int
	verbose      bool
	accelCache   map[string][]int
}

type KnobDef struct {
	name     string
	getter   func(c *Candidate) int
	setter   func(c *Candidate, v int)
	min, max int
}

func buildKnobs() []KnobDef {
	knobs := make([]KnobDef, 0, 19)
	for op := 0; op < 6; op++ {
		opIdx := op
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Level", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].OutputLevel) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].OutputLevel = uint8(v) },
			min:    0, max: 99,
		})
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Fine", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].FreqFine) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].FreqFine = uint8(v) },
			min:    0, max: 99,
		})
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Coarse", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].FreqCoarse) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].FreqCoarse = uint8(v) },
			min:    0, max: 31,
		})
	}
	knobs = append(knobs, KnobDef{
		name:   "Feedback",
		getter: func(c *Candidate) int { return int(c.Feedback) },
		setter: func(c *Candidate, v int) { c.Feedback = uint8(v) },
		min:    0, max: 7,
	})
	knobs = append(knobs, KnobDef{
		name:   "Pitch.Level1",
		getter: func(c *Candidate) int { return int(c.PitchLevel1) },
		setter: func(c *Candidate, v int) { c.PitchLevel1 = uint8(v) },
		min:    0, max: 99,
	})
	knobs = append(knobs, KnobDef{
		name:   "Pitch.Level2",
		getter: func(c *Candidate) int { return int(c.PitchLevel2) },
		setter: func(c *Candidate, v int) { c.PitchLevel2 = uint8(v) },
		min:    0, max: 99,
	})
	knobs = append(knobs, KnobDef{
		name:   "Pitch.Rate1",
		getter: func(c *Candidate) int { return int(c.PitchRate1) },
		setter: func(c *Candidate, v int) { c.PitchRate1 = uint8(v) },
		min:    0, max: 99,
	})
	return knobs
}

func NewLiveOptimizer(target []float64, initial *Candidate, maxIter int, verbose bool, note uint8, dur time.Duration) *LiveOptimizer {
	targetNorm := NormalizeRMS(target)
	optDur := dur

	return &LiveOptimizer{
		currentPatch: initial.Copy(),
		targetNorm:   targetNorm,
		note:         note,
		dur:          optDur,
		bestDist:     math.MaxFloat64,
		bestPatch:    initial.Copy(),
		maxIter:      maxIter,
		verbose:      verbose,
		accelCache:   make(map[string][]int),
	}
}

// GeneratePermutationCandidate generates a candidate patch by sampling parameter values
// from the cached acceleration troughs discovered during live optimization.
func (lo *LiveOptimizer) GeneratePermutationCandidate(rng *rand.Rand, algo uint8) *Candidate {
	c := lo.bestPatch.Copy()
	c.Algorithm = algo

	if len(lo.accelCache) == 0 {
		return c
	}

	knobs := buildKnobs()
	for _, k := range knobs {
		troughs, ok := lo.accelCache[k.name]
		if ok && len(troughs) > 0 {
			v := troughs[rng.Intn(len(troughs))]
			if v >= k.min && v <= k.max {
				k.setter(c, v)
			}
		}
	}
	return c
}

func (lo *LiveOptimizer) measureDeterministic(patch *Candidate) float64 {
	candAudio := RenderCandidate(patch, lo.note, lo.dur)
	candidate := TrimOrPad(candAudio, len(lo.targetNorm))
	candidate = NormalizeRMS(candidate)
	return SpectralDistance(lo.targetNorm, candidate)
}

type candidateMeta struct {
	knobName string
	val      int
}

type accelerationEntry struct {
	val   int
	accel float64
}

// updateAccelerationCache computes discrete 2nd derivative acceleration scores
// for knob sweep results and stores the top 3 acceleration troughs per knob.
func updateAccelerationCache(cache map[string][]int, knobs []KnobDef, baseDist float64, cands []*Candidate, meta []candidateMeta, fits []float64) {
	type knobStep struct {
		val int
		fit float64
	}
	knobData := make(map[string][]knobStep)

	for i, m := range meta {
		knobData[m.knobName] = append(knobData[m.knobName], knobStep{val: m.val, fit: fits[i]})
	}

	for kName, steps := range knobData {
		if len(steps) < 3 {
			continue
		}
		sort.Slice(steps, func(i, j int) bool { return steps[i].val < steps[j].val })

		var troughs []accelerationEntry
		for i := 1; i < len(steps)-1; i++ {
			dPrev := steps[i-1].fit
			dCur := steps[i].fit
			dNext := steps[i+1].fit

			// Acceleration (discrete 2nd derivative): dPrev - 2*dCur + dNext
			accel := dPrev - 2.0*dCur + dNext
			if dCur < baseDist && accel > 1e-4 {
				troughs = append(troughs, accelerationEntry{val: steps[i].val, accel: accel})
			}
		}

		sort.Slice(troughs, func(i, j int) bool { return troughs[i].accel > troughs[j].accel })

		var topVals []int
		for rank := 0; rank < 3 && rank < len(troughs); rank++ {
			topVals = append(topVals, troughs[rank].val)
		}
		if len(topVals) > 0 {
			cache[kName] = topVals
		}
	}
}

// generateProbeBatch creates candidate patches testing cached acceleration troughs.
func generateProbeBatch(base *Candidate, knobs []KnobDef, cache map[string][]int) ([]*Candidate, []candidateMeta) {
	var cands []*Candidate
	var meta []candidateMeta

	for _, k := range knobs {
		troughs, ok := cache[k.name]
		if !ok || len(troughs) == 0 {
			continue
		}
		curVal := k.getter(base)
		for _, v := range troughs {
			if v != curVal && v >= k.min && v <= k.max {
				c := base.Copy()
				k.setter(c, v)
				cands = append(cands, c)
				meta = append(meta, candidateMeta{knobName: k.name, val: v})
			}
		}
	}
	return cands, meta
}

func generateBatch(base *Candidate, knobs []KnobDef, fine bool) ([]*Candidate, []candidateMeta) {
	var cands []*Candidate
	var meta []candidateMeta

	for _, k := range knobs {
		curVal := k.getter(base)
		if fine {
			for delta := -3; delta <= 3; delta++ {
				if delta == 0 {
					continue
				}
				v := curVal + delta
				if v >= k.min && v <= k.max {
					c := base.Copy()
					k.setter(c, v)
					cands = append(cands, c)
					meta = append(meta, candidateMeta{knobName: k.name, val: v})
				}
			}
		} else {
			step := max(1, (k.max-k.min)/20)
			for v := k.min; v <= k.max; v += step {
				if v == curVal {
					continue
				}
				c := base.Copy()
				k.setter(c, v)
				cands = append(cands, c)
				meta = append(meta, candidateMeta{knobName: k.name, val: v})
			}
		}
	}
	return cands, meta
}

func (lo *LiveOptimizer) Run() *Candidate {
	knobs := buildKnobs()

	initialDist := lo.measureDeterministic(lo.currentPatch)
	lo.bestDist = initialDist
	lo.bestPatch = lo.currentPatch.Copy()

	pianoPatch := DecodeVoicePatch(epiano)
	pianoDist := lo.measureDeterministic(pianoPatch)

	if lo.verbose {
		fmt.Printf("[live-parallel] Starting: Piano1(%.6f) -> candidate(%.6f), 19 knobs\n", pianoDist, initialDist)
	}

	if lo.bestDist <= 1e-4 {
		if lo.verbose {
			fmt.Printf("[live] Candidate is already an exact match (dist=%.6f) — skipping live optimization\n", lo.bestDist)
		}
		return lo.bestPatch
	}

	type patchSeed struct {
		patch *Candidate
		label string
	}

	var seeds []patchSeed
	initialSeeds := []*Candidate{lo.currentPatch.Copy(), pianoPatch}

	// 1. Initial 32-Algorithm Sweep for each seed to discover top algorithm topologies
	for sIdx, seed := range initialSeeds {
		seedDist := lo.measureDeterministic(seed)
		// Skip Piano 1 base if current best candidate is already significantly better
		if sIdx == 1 && seedDist >= lo.bestDist+0.2 {
			if lo.verbose {
				fmt.Printf("[live-parallel] Skipping Piano 1 base (piano dist %.6f >= best %.6f + 0.2)\n", seedDist, lo.bestDist)
			}
			continue
		}

		algoCands := make([]*Candidate, 32)
		for a := uint8(0); a < 32; a++ {
			c := seed.Copy()
			c.Algorithm = a
			algoCands[a] = c
		}

		fits, _ := evaluateBatchParallel(algoCands, lo.targetNorm, lo.note, lo.dur)

		type algoEntry struct {
			cand *Candidate
			fit  float64
		}
		entries := make([]algoEntry, 32)
		for i := 0; i < 32; i++ {
			entries[i] = algoEntry{cand: algoCands[i], fit: fits[i]}
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].fit < entries[j].fit })

		seedName := "Candidate"
		if sIdx == 1 {
			seedName = "Piano1"
		}

		// Take the best algorithm topology from the sweep
		if len(entries) > 0 {
			c := entries[0].cand.Copy()
			f := entries[0].fit

			if f < lo.bestDist {
				lo.bestDist = f
				lo.bestPatch = c.Copy()
			}

			label := fmt.Sprintf("%s (Alg %d, dist=%.6f)", seedName, c.Algorithm+1, f)
			seeds = append(seeds, patchSeed{patch: c, label: label})
		}
	}

	// 2. Multi-Algorithm Branch Optimization
	for _, seed := range seeds {
		curPatch := seed.patch.Copy()
		curDist := lo.measureDeterministic(curPatch)

		// Per-branch local acceleration cache
		accelCache := make(map[string][]int)

		// Prune branch if its starting distance is significantly worse than overall best
		if curDist >= lo.bestDist+0.25 {
			if lo.verbose {
				fmt.Printf("[live-parallel] Pruning branch %s (dist %.6f >= best %.6f + 0.25)\n",
					seed.label, curDist, lo.bestDist)
			}
			continue
		}

		if curDist < lo.bestDist {
			lo.bestDist = curDist
			lo.bestPatch = curPatch.Copy()
		}

		if lo.verbose {
			fmt.Printf("[live-parallel] Optimizing branch %s\n", seed.label)
		}

		maxBatchIters := lo.maxIter
		if maxBatchIters > 10 {
			maxBatchIters = 10
		}

		for iter := 0; iter < maxBatchIters; iter++ {
			// A. Acceleration Trough Probe Pass (ultra-fast leaps to cached 2nd-derivative troughs)
			probeCands, probeMeta := generateProbeBatch(curPatch, knobs, accelCache)
			if len(probeCands) > 0 {
				fits, _ := evaluateBatchParallel(probeCands, lo.targetNorm, lo.note, lo.dur)

				bestIdx := -1
				bestFit := curDist
				for i, f := range fits {
					if f < bestFit {
						bestFit = f
						bestIdx = i
					}
				}

				if bestIdx >= 0 && (curDist-bestFit) > 1e-4 {
					prevDist := curDist
					curDist = bestFit
					curPatch = probeCands[bestIdx].Copy()
					m := probeMeta[bestIdx]

					if curDist < lo.bestDist {
						lo.bestDist = curDist
						lo.bestPatch = curPatch.Copy()
					}

					if lo.verbose {
						fmt.Printf("[live-accel] iter %d: %s -> %d (dist=%.6f, leaped via acceleration trough from %.6f!)\n",
							iter, m.knobName, m.val, curDist, prevDist)
					}
					continue
				}
			}

			// B. Coarse pass
			coarseCands, coarseMeta := generateBatch(curPatch, knobs, false)
			if len(coarseCands) > 0 {
				fits, _ := evaluateBatchParallel(coarseCands, lo.targetNorm, lo.note, lo.dur)

				bestIdx := -1
				bestFit := curDist
				for i, f := range fits {
					if f < bestFit {
						bestFit = f
						bestIdx = i
					}
				}

				if bestIdx >= 0 && (curDist-bestFit) > 1e-4 {
					prevDist := curDist
					curDist = bestFit

					// Record discrete 2nd-derivative acceleration troughs for future probe leaps
					updateAccelerationCache(accelCache, knobs, prevDist, coarseCands, coarseMeta, fits)
					lo.accelCache = accelCache

					curPatch = coarseCands[bestIdx].Copy()
					m := coarseMeta[bestIdx]

					if curDist < lo.bestDist {
						lo.bestDist = curDist
						lo.bestPatch = curPatch.Copy()
					}

					if lo.verbose {
						fmt.Printf("[live] iter %d: %s -> %d (dist=%.6f, improved from %.6f)\n",
							iter, m.knobName, m.val, curDist, prevDist)
					}
					continue
				}
			}

			// C. Fine pass
			fineCands, fineMeta := generateBatch(curPatch, knobs, true)
			if len(fineCands) > 0 {
				fineFits, _ := evaluateBatchParallel(fineCands, lo.targetNorm, lo.note, lo.dur)

				fineBestIdx := -1
				fineBestFit := curDist
				for i, f := range fineFits {
					if f < fineBestFit {
						fineBestFit = f
						fineBestIdx = i
					}
				}

				if fineBestIdx >= 0 && (curDist-fineBestFit) > 1e-4 {
					prevDist := curDist
					curDist = fineBestFit
					curPatch = fineCands[fineBestIdx].Copy()
					m := fineMeta[fineBestIdx]

					if curDist < lo.bestDist {
						lo.bestDist = curDist
						lo.bestPatch = curPatch.Copy()
					}

					if lo.verbose {
						fmt.Printf("[live-fine] iter %d: %s -> %d (dist=%.6f, improved from %.6f)\n",
							iter, m.knobName, m.val, curDist, prevDist)
					}
					continue
				}
			}

			// D. Topological algorithm jump guarantee on exit or plateau
			algoCands := make([]*Candidate, 32)
			for a := uint8(0); a < 32; a++ {
				c := curPatch.Copy()
				c.Algorithm = a
				algoCands[a] = c
			}

			algoFits, _ := evaluateBatchParallel(algoCands, lo.targetNorm, lo.note, lo.dur)

			algoBestIdx := -1
			algoBestFit := curDist
			for i, f := range algoFits {
				if f < algoBestFit {
					algoBestFit = f
					algoBestIdx = i
				}
			}

			if algoBestIdx >= 0 && (curDist-algoBestFit) > 1e-4 {
				prevDist := curDist
				prevAlgo := curPatch.Algorithm
				curDist = algoBestFit
				curPatch = algoCands[algoBestIdx].Copy()

				// Wipe acceleration cache on algorithm topology change
				accelCache = make(map[string][]int)

				if curDist < lo.bestDist {
					lo.bestDist = curDist
					lo.bestPatch = curPatch.Copy()
				}

				if lo.verbose {
					fmt.Printf("[live-algo] iter %d: Algorithm %d -> %d (dist=%.6f, improved from %.6f)\n",
						iter, prevAlgo+1, curPatch.Algorithm+1, curDist, prevDist)
				}
				continue
			}

			if lo.verbose {
				fmt.Printf("[live] Local minimum reached for %s at dist=%.6f\n", seed.label, curDist)
			}
			break
		}
	}

	if lo.verbose {
		fmt.Printf("[live] Final Best distance: %.6f\n", lo.bestDist)
	}

	return lo.bestPatch
}

// BestDist returns the best fitness distance found by the LiveOptimizer.
func (lo *LiveOptimizer) BestDist() float64 {
	return lo.bestDist
}
