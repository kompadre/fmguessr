package match

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"
)

type LiveOptimizer struct {
	currentPatch *Candidate
	targetNorm   []float64
	rawTargetRMS float64
	note         uint8
	dur          time.Duration
	bestDist     float64
	bestPatch    *Candidate
	maxIter      int
	verbose      bool
	accelCache   map[string][]int
	ctx          context.Context
	targetMel    *PrecomputedTargetMel
}

type KnobDef struct {
	name     string
	getter   func(c *Candidate) int
	setter   func(c *Candidate, v int)
	min, max int
}

func buildKnobs(includePitch bool) []KnobDef {
	knobs := make([]KnobDef, 0, 22)
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
	if includePitch {
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
	}
	return knobs
}

func NewLiveOptimizer(ctx context.Context, target []float64, initial *Candidate, maxIter int, verbose bool, note uint8, dur time.Duration, rawTargetRMS ...float64) *LiveOptimizer {
	targetNorm := NormalizeRMS(target)
	optDur := dur
	targetRMS := 0.0
	if len(rawTargetRMS) > 0 {
		if rawTargetRMS[0] > 0 {
			targetRMS = rawTargetRMS[0]
		}
	} else {
		targetRMS = CalcRMS(target)
	}

	return &LiveOptimizer{
		currentPatch: initial.Copy(),
		targetNorm:   targetNorm,
		rawTargetRMS: targetRMS,
		note:         note,
		dur:          optDur,
		bestDist:     math.MaxFloat64,
		bestPatch:    initial.Copy(),
		maxIter:      maxIter,
		verbose:      verbose,
		accelCache:   make(map[string][]int),
		ctx:          ctx,
	}
}

// WithTargetMel sets the cached target mel spectrogram for faster distance evaluation.
func (lo *LiveOptimizer) WithTargetMel(tm *PrecomputedTargetMel) *LiveOptimizer {
	lo.targetMel = tm
	return lo
}

// GeneratePermutationCandidate generates a candidate patch by sampling parameter values
// from the cached acceleration troughs discovered during live optimization.
func (lo *LiveOptimizer) GeneratePermutationCandidate(rng *rand.Rand, algo uint8) *Candidate {
	c := lo.bestPatch.Copy()
	c.Algorithm = algo

	if len(lo.accelCache) == 0 {
		return c
	}

	knobs := buildKnobs(true)
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
	trimmed := TrimOrPad(candAudio, len(lo.targetNorm))
	rawCandRMS := CalcRMS(trimmed)
	candidate := NormalizeRMS(trimmed)
	var dist float64
	if lo.targetMel != nil {
		dist = MelCosineDistanceCached(lo.targetMel, candidate, RenderSampleRate)
	} else {
		dist = MelCosineDistance(lo.targetNorm, candidate, RenderSampleRate)
	}
	if lo.rawTargetRMS > 1e-4 {
		dist += CalcDBPenalty(rawCandRMS, lo.rawTargetRMS)
	}
	return dist
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

type knobDelta struct {
	knobIdx int
	knob    KnobDef
	delta   int
	imp     float64
}

// generateMultiKnobTrajectoryCandidates combines top single-knob improvers into scaled joint shift vectors.
func generateMultiKnobTrajectoryCandidates(base *Candidate, knobs []KnobDef, improvers []knobDelta) ([]*Candidate, []candidateMeta) {
	if len(improvers) < 2 {
		return nil, nil
	}

	sort.Slice(improvers, func(i, j int) bool { return improvers[i].imp > improvers[j].imp })

	if len(improvers) > 6 {
		improvers = improvers[:6]
	}

	scales := []float64{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0}
	cands := make([]*Candidate, 0, len(scales))
	meta := make([]candidateMeta, 0, len(scales))

	for _, s := range scales {
		c := base.Copy()
		var applied []string
		for _, imp := range improvers {
			scaledDelta := int(math.Round(float64(imp.delta) * s))
			if scaledDelta == 0 {
				continue
			}
			curVal := imp.knob.getter(c)
			newVal := curVal + scaledDelta
			if newVal < imp.knob.min {
				newVal = imp.knob.min
			}
			if newVal > imp.knob.max {
				newVal = imp.knob.max
			}
			imp.knob.setter(c, newVal)
			applied = append(applied, fmt.Sprintf("%s%+d", imp.knob.name, newVal-curVal))
		}
		if len(applied) > 0 {
			cands = append(cands, c)
			label := fmt.Sprintf("Trajectory[%.2fx: %s]", s, strings.Join(applied, ","))
			meta = append(meta, candidateMeta{knobName: label, val: 0})
		}
	}

	return cands, meta
}

// generateCoupledOperatorCandidates sweeps FM operator pairs (Levels and Coarse/Fine) simultaneously.
func generateCoupledOperatorCandidates(base *Candidate, knobs []KnobDef) ([]*Candidate, []candidateMeta) {
	var cands []*Candidate
	var meta []candidateMeta

	algo := base.Algorithm

	// 1. Coupled Modulator-Carrier Level Sweeps
	levelDeltas := [][2]int{
		{2, 2}, {4, 2}, {2, 4}, {4, -2}, {-2, 4}, {-3, -3}, {6, 3}, {-4, -2},
	}

	for mOp := 0; mOp < 6; mOp++ {
		if IsCarrier(algo, mOp) {
			continue
		}
		for cOp := 0; cOp < 6; cOp++ {
			if !IsCarrier(algo, cOp) || mOp == cOp {
				continue
			}
			mLvl := int(base.Ops[mOp].OutputLevel)
			cLvl := int(base.Ops[cOp].OutputLevel)

			for _, d := range levelDeltas {
				newM := mLvl + d[0]
				newC := cLvl + d[1]
				if newM >= 0 && newM <= 99 && newC >= 0 && newC <= 99 {
					c := base.Copy()
					c.Ops[mOp].OutputLevel = uint8(newM)
					c.Ops[cOp].OutputLevel = uint8(newC)
					cands = append(cands, c)
					label := fmt.Sprintf("CoupledLvl[Op%d%+d,Op%d%+d]", mOp+1, d[0], cOp+1, d[1])
					meta = append(meta, candidateMeta{knobName: label, val: newM})
				}
			}
		}
	}

	// 2. Coupled Coarse & Fine Frequency Sweeps per operator
	freqDeltas := [][2]int{
		{1, -5}, {1, 5}, {1, -10}, {1, 10}, {-1, 10}, {-1, -10}, {2, -10},
	}

	for op := 0; op < 6; op++ {
		coarse := int(base.Ops[op].FreqCoarse)
		fine := int(base.Ops[op].FreqFine)

		for _, d := range freqDeltas {
			newC := coarse + d[0]
			newF := fine + d[1]
			if newC >= 0 && newC <= 31 && newF >= 0 && newF <= 99 {
				c := base.Copy()
				c.Ops[op].FreqCoarse = uint8(newC)
				c.Ops[op].FreqFine = uint8(newF)
				cands = append(cands, c)
				label := fmt.Sprintf("CoupledFreq[Op%d.C%+d,F%+d]", op+1, d[0], d[1])
				meta = append(meta, candidateMeta{knobName: label, val: newC})
			}
		}
	}

	return cands, meta
}

// generateMultiDialCrossTroughCandidates samples multi-dial combinations from cached acceleration troughs.
func generateMultiDialCrossTroughCandidates(base *Candidate, knobs []KnobDef, cache map[string][]int, rng *rand.Rand) ([]*Candidate, []candidateMeta) {
	if len(cache) < 2 {
		return nil, nil
	}

	type activeKnob struct {
		knob    KnobDef
		troughs []int
	}

	var active []activeKnob
	for _, k := range knobs {
		if t, ok := cache[k.name]; ok && len(t) > 0 {
			active = append(active, activeKnob{knob: k, troughs: t})
		}
	}

	if len(active) < 2 {
		return nil, nil
	}

	var cands []*Candidate
	var meta []candidateMeta

	numPerms := 25
	for p := 0; p < numPerms; p++ {
		c := base.Copy()
		maxD := len(active)
		if maxD > 4 {
			maxD = 4
		}
		numDials := 2
		if maxD > 2 {
			numDials = 2 + rng.Intn(maxD-1)
		}
		perm := rng.Perm(len(active))
		var applied []string

		for i := 0; i < numDials; i++ {
			ak := active[perm[i]]
			val := ak.troughs[rng.Intn(len(ak.troughs))]
			curVal := ak.knob.getter(c)
			if val != curVal && val >= ak.knob.min && val <= ak.knob.max {
				ak.knob.setter(c, val)
				applied = append(applied, fmt.Sprintf("%s->%d", ak.knob.name, val))
			}
		}

		if len(applied) > 0 {
			cands = append(cands, c)
			label := fmt.Sprintf("CrossTrough[%s]", strings.Join(applied, ","))
			meta = append(meta, candidateMeta{knobName: label, val: 0})
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
	knobs := buildKnobs(true)

	initialDist := lo.measureDeterministic(lo.currentPatch)
	lo.bestDist = initialDist
	lo.bestPatch = lo.currentPatch.Copy()

	if lo.bestDist <= 1e-4 {
		if lo.verbose {
			fmt.Printf("%dms] [live] Candidate is already an exact match (dist=%.6f) — skipping live optimization\n", logMs(), lo.bestDist)
		}
		return lo.bestPatch
	}

	// When GA has already converged well (dist < 0.05), skip multi-seed
	// exploration. Waveform seeds (Sine/Square/etc) at dist 0.1+ are a waste.
	// Just refine the GA best candidate directly.
	skipSeeds := lo.bestDist < 0.05

	type seedDef struct {
		patch *Candidate
		name  string
	}

	var seedDefs []seedDef
	if skipSeeds {
		seedDefs = []seedDef{
			{patch: lo.currentPatch.Copy(), name: "Candidate"},
		}
		if lo.verbose {
			fmt.Printf("%dms] [live-parallel] Starting: candidate(%.6f), 19 knobs (skipping waveform seeds — GA converged)\n", logMs(), initialDist)
		}
	} else {
		pianoPatch := DecodeVoicePatch(epiano)
		pianoDist := lo.measureDeterministic(pianoPatch)
		seedDefs = []seedDef{
			{patch: lo.currentPatch.Copy(), name: "Candidate"},
			{patch: SineBase(), name: "Sine"},
			{patch: SquareBase(), name: "Square"},
			{patch: SawtoothBase(), name: "Sawtooth"},
			{patch: TriangleBase(), name: "Triangle"},
			{patch: pianoPatch, name: "Piano1"},
		}
		if lo.verbose {
			fmt.Printf("%dms] [live-parallel] Starting: Piano1(%.6f) -> candidate(%.6f), 19 knobs\n", logMs(), pianoDist, initialDist)
		}
	}

	type patchSeed struct {
		patch *Candidate
		label string
	}

	var seeds []patchSeed

	// 1. Initial 32-Algorithm Sweep for each pure waveform seed to discover top algorithm topologies
	for sIdx, sd := range seedDefs {
		seedDist := lo.measureDeterministic(sd.patch)
		// Skip non-candidate seed if current best candidate is already significantly better
		if sIdx > 0 && seedDist >= lo.bestDist+0.25 {
			if lo.verbose {
				fmt.Printf("%dms] [live-parallel] Skipping %s base (dist %.6f >= best %.6f + 0.25)\n", logMs(), sd.name, seedDist, lo.bestDist)
			}
			continue
		}

		algoCands := make([]*Candidate, 32)
		for a := uint8(0); a < 32; a++ {
			c := sd.patch.Copy()
			c.Algorithm = a
			algoCands[a] = c
		}

		fits, _ := evaluateBatchParallel(lo.ctx, algoCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)

		type algoEntry struct {
			cand *Candidate
			fit  float64
		}
		entries := make([]algoEntry, 32)
		for i := 0; i < 32; i++ {
			entries[i] = algoEntry{cand: algoCands[i], fit: fits[i]}
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].fit < entries[j].fit })

		// Take the best algorithm topology from the sweep
		if len(entries) > 0 {
			c := entries[0].cand.Copy()
			f := entries[0].fit

			if f < lo.bestDist {
				lo.bestDist = f
				lo.bestPatch = c.Copy()
			}

			label := fmt.Sprintf("%s (Alg %d, dist=%.6f)", sd.name, c.Algorithm+1, f)
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
				fmt.Printf("%dms] [live-parallel] Pruning branch %s (dist %.6f >= best %.6f + 0.25)\n", logMs(),
					seed.label, curDist, lo.bestDist)
			}
			continue
		}

		if curDist < lo.bestDist {
			lo.bestDist = curDist
			lo.bestPatch = curPatch.Copy()
		}

		if lo.verbose {
			fmt.Printf("%dms] [live-parallel] Optimizing branch %s\n", logMs(), seed.label)
		}

		maxBatchIters := lo.maxIter
		if maxBatchIters > 10 {
			maxBatchIters = 10
		}

	branchLoop:
		for iter := 0; iter < maxBatchIters; iter++ {
			// Check context cancellation
			select {
			case <-lo.ctx.Done():
				if lo.verbose {
					fmt.Printf("%dms] [live] CANCELLED at iter %d — stopping branch %s\n", logMs(), iter, seed.label)
				}
				break branchLoop
			default:
			}

			// A. Acceleration Trough Probe Pass (ultra-fast leaps to cached 2nd-derivative troughs)
			// A. Acceleration Trough Probe Pass (ultra-fast leaps to cached 2nd-derivative troughs)
			probeCands, probeMeta := generateProbeBatch(curPatch, knobs, accelCache)
			if len(probeCands) > 0 {
				fits, _ := evaluateBatchParallel(lo.ctx, probeCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)

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
						fmt.Printf("%dms] [live-accel] iter %d: %s -> %d (dist=%.6f, leaped via acceleration trough from %.6f!)\n", logMs(),
							iter, m.knobName, m.val, curDist, prevDist)
					}
					continue
				}
			}

			// B. Coarse pass (1D single knob sweeps + Multi-Knob Coordinate Vector & Coupled Operator Sweeps)
			coarseCands, coarseMeta := generateBatch(curPatch, knobs, false)
			if len(coarseCands) > 0 {
				fits, _ := evaluateBatchParallel(lo.ctx, coarseCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)

				bestIdx := -1
				bestFit := curDist
				var improvers []knobDelta

				for i, f := range fits {
					if f < bestFit {
						bestFit = f
						bestIdx = i
					}
					if (curDist - f) > 1e-4 {
						m := coarseMeta[i]
						for kIdx, k := range knobs {
							if k.name == m.knobName {
								curVal := k.getter(curPatch)
								improvers = append(improvers, knobDelta{
									knobIdx: kIdx,
									knob:    k,
									delta:   m.val - curVal,
									imp:     curDist - f,
								})
								break
							}
						}
					}
				}

				// Check Multi-Knob Coordinate Trajectory & Coupled Operator Sweeps
				var multiCands []*Candidate
				var multiMeta []candidateMeta

				if len(improvers) >= 2 {
					trajCands, trajMeta := generateMultiKnobTrajectoryCandidates(curPatch, knobs, improvers)
					multiCands = append(multiCands, trajCands...)
					multiMeta = append(multiMeta, trajMeta...)
				}

				coupledCands, coupledMeta := generateCoupledOperatorCandidates(curPatch, knobs)
				multiCands = append(multiCands, coupledCands...)
				multiMeta = append(multiMeta, coupledMeta...)

				numSingleKnob := len(coarseCands)
				if len(multiCands) > 0 {
					mFits, _ := evaluateBatchParallel(lo.ctx, multiCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)
					for i, mf := range mFits {
						if mf < bestFit {
							bestFit = mf
							bestIdx = numSingleKnob + i
						}
					}
					coarseCands = append(coarseCands, multiCands...)
					coarseMeta = append(coarseMeta, multiMeta...)
					fits = append(fits, mFits...)
				}

				if bestIdx >= 0 && (curDist-bestFit) > 1e-4 {
					prevDist := curDist
					curDist = bestFit

					// Record discrete 2nd-derivative acceleration troughs for future probe leaps
					updateAccelerationCache(accelCache, knobs, prevDist, coarseCands[:numSingleKnob], coarseMeta[:numSingleKnob], fits[:numSingleKnob])
					lo.accelCache = accelCache

					curPatch = coarseCands[bestIdx].Copy()
					m := coarseMeta[bestIdx]

					if curDist < lo.bestDist {
						lo.bestDist = curDist
						lo.bestPatch = curPatch.Copy()
					}

					if lo.verbose {
						fmt.Printf("%dms] [live] iter %d: %s (dist=%.6f, improved from %.6f)\n", logMs(),
							iter, m.knobName, curDist, prevDist)
					}
					continue
				}
			}

			// C. Fine pass
			fineCands, fineMeta := generateBatch(curPatch, knobs, true)
			if len(fineCands) > 0 {
				fineFits, _ := evaluateBatchParallel(lo.ctx, fineCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)

				fineBestIdx := -1
				fineBestFit := curDist
				var fineImprovers []knobDelta

				for i, f := range fineFits {
					if f < fineBestFit {
						fineBestFit = f
						fineBestIdx = i
					}
					if (curDist - f) > 1e-4 {
						m := fineMeta[i]
						for kIdx, k := range knobs {
							if k.name == m.knobName {
								curVal := k.getter(curPatch)
								fineImprovers = append(fineImprovers, knobDelta{
									knobIdx: kIdx,
									knob:    k,
									delta:   m.val - curVal,
									imp:     curDist - f,
								})
								break
							}
						}
					}
				}

				// Check Multi-Knob Fine Trajectories & Multi-Dial Cross-Troughs
				var fineMultiCands []*Candidate
				var fineMultiMeta []candidateMeta

				if len(fineImprovers) >= 2 {
					fineTrajCands, fineTrajMeta := generateMultiKnobTrajectoryCandidates(curPatch, knobs, fineImprovers)
					fineMultiCands = append(fineMultiCands, fineTrajCands...)
					fineMultiMeta = append(fineMultiMeta, fineTrajMeta...)
				}

				rng := rand.New(rand.NewSource(time.Now().UnixNano()))
				crossCands, crossMeta := generateMultiDialCrossTroughCandidates(curPatch, knobs, accelCache, rng)
				fineMultiCands = append(fineMultiCands, crossCands...)
				fineMultiMeta = append(fineMultiMeta, crossMeta...)

				numSingleFine := len(fineCands)
				if len(fineMultiCands) > 0 {
					fmFits, _ := evaluateBatchParallel(lo.ctx, fineMultiCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)
					for i, fmf := range fmFits {
						if fmf < fineBestFit {
							fineBestFit = fmf
							fineBestIdx = numSingleFine + i
						}
					}
					fineCands = append(fineCands, fineMultiCands...)
					fineMeta = append(fineMeta, fineMultiMeta...)
					fineFits = append(fineFits, fmFits...)
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
						fmt.Printf("%dms] [live-fine] iter %d: %s (dist=%.6f, improved from %.6f)\n", logMs(),
							iter, m.knobName, curDist, prevDist)
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

			algoFits, _ := evaluateBatchParallel(lo.ctx, algoCands, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, nil)

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
					fmt.Printf("%dms] [live-algo] iter %d: Algorithm %d -> %d (dist=%.6f, improved from %.6f)\n", logMs(),
						iter, prevAlgo+1, curPatch.Algorithm+1, curDist, prevDist)
				}
				continue
			}

			// E. Operator Block Mini-GA Crossbreeding on Plateau
			miniGABest := RunOperatorMiniGA(lo.ctx, []*Candidate{curPatch, lo.bestPatch, SineBase(), SquareBase(), SawtoothBase(), TriangleBase()}, lo.targetNorm, lo.note, lo.dur, lo.rawTargetRMS, 10, 16, lo.verbose)
			if miniGABest != nil && (curDist-miniGABest.Fitness) > 1e-4 {
				prevDist := curDist
				curDist = miniGABest.Fitness
				curPatch = miniGABest.Copy()
				accelCache = make(map[string][]int) // wipe cache on hybrid crossbreed

				if curDist < lo.bestDist {
					lo.bestDist = curDist
					lo.bestPatch = curPatch.Copy()
				}

				if lo.verbose {
					fmt.Printf("%dms] [live-miniGA] iter %d: Operator Hybrid Crossbreed (dist=%.6f, improved from %.6f)\n", logMs(),
						iter, curDist, prevDist)
				}
				continue
			}

			if lo.verbose {
				fmt.Printf("%dms] [live] Local minimum reached for %s at dist=%.6f\n", logMs(), seed.label, curDist)
			}
			break
		}
	}

	if lo.verbose {
		fmt.Printf("%dms] [live] Final Best distance: %.6f\n", logMs(), lo.bestDist)
	}

	return lo.bestPatch
}

// BestDist returns the best fitness distance found by the LiveOptimizer.
func (lo *LiveOptimizer) BestDist() float64 {
	return lo.bestDist
}
