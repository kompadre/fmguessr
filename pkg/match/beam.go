package match

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"
)

// OperatorAddition records one step in the constructive evolution.
type OperatorAddition struct {
	Step        int       // 0-5
	OpIndex     int       // DX7 operator index (0-5), placed in reverse: 5,4,3,2,1,0
	Role        string    // "carrier" or "modulator"
	Algorithm   uint8     // algorithm after this addition
	ModTarget   int       // if modulator, which op it modulates (-1 if carrier)
	Params      OpParams  // snapshot of the new operator's params
	Feedback    uint8     // feedback level at this step
	Distance    float64   // spectral distance after this step
	Description string    // human-readable summary
}

// PatchEvolution is the output artifact of the constructive beam search.
type PatchEvolution struct {
	Final        Candidate
	Trace        []OperatorAddition
	Algorithm    uint8
	BeamWidth    int
	TotalRenders int
}

// beamCandidate is working state during search.
type beamCandidate struct {
	candidate  Candidate
	activeOps  []int           // operator indices in placement order
	roles      map[int]string  // opIndex -> "carrier"/"modulator"
	modTargets map[int]int     // opIndex -> target opIndex (for modulators)
	distance   float64
}

// algoRole describes one valid (algorithm, role, target) combination for a new operator.
type algoRole struct {
	algo    uint8
	role    string // "carrier" or "modulator"
	target  int    // modulator target opIndex, -1 for carrier
}

// opRouting precomputed routing info for one operator in one algorithm.
type opRouting struct {
	isCarrier   bool
	outBus      int // 0=final, 1=bus1, 2=bus2
	inBus       int // 0=none, 1=bus1, 2=bus2
	hasFeedback bool
}

// algoRoutingTable precomputed for all 32 algorithms × 6 operators.
var algoRoutingTable [32][6]opRouting

func init() {
	for algo := 0; algo < 32; algo++ {
		for opIdx := 0; opIdx < 6; opIdx++ {
			flags := algOps[algo][opIdx]
			algoRoutingTable[algo][opIdx] = opRouting{
				isCarrier:   (flags & 3) == 0,
				outBus:      int(flags & 3),
				inBus:       int((flags >> 4) & 3),
				hasFeedback: (flags & 0xc0) == 0xc0,
			}
		}
	}
}

// placementOrder is the fixed reverse-index operator placement sequence.
var placementOrder = [6]int{5, 4, 3, 2, 1, 0}

// getAlgorithmsForConfig returns all valid (algorithm, role, target) triples
// for adding a new operator given the current active configuration.
func getAlgorithmsForConfig(activeRoles map[int]string, newOpIdx int) []algoRole {
	var results []algoRole

	for algo := 0; algo < 32; algo++ {
		routing := algoRoutingTable[algo]
		newRoute := routing[newOpIdx]

		// Check that all existing active ops are compatible with this algorithm
		compatible := true
		for opIdx, role := range activeRoles {
			r := routing[opIdx]
			if role == "carrier" && !r.isCarrier {
				compatible = false
				break
			}
			if role == "modulator" && r.isCarrier {
				compatible = false
				break
			}
		}
		if !compatible {
			continue
		}

		// Try new op as carrier
		if newRoute.isCarrier {
			results = append(results, algoRole{
				algo:   uint8(algo),
				role:   "carrier",
				target: -1,
			})
		}

		// Try new op as modulator: find active operators that write to its input bus
		if !newRoute.isCarrier && newRoute.inBus != 0 {
			for targetOp := range activeRoles {
				targetRoute := routing[targetOp]
				// Target must write to the bus the new op reads from
				if targetRoute.outBus == newRoute.inBus {
					results = append(results, algoRole{
						algo:   uint8(algo),
						role:   "modulator",
						target: targetOp,
					})
				}
			}
		}
	}

	return results
}

// prepareCandidate creates a candidate from a beam state with the new operator
// configured and all inactive operators muted.
func prepareCandidate(bc *beamCandidate, algo uint8, newOpIdx int, params OpParams, feedback uint8) Candidate {
	cand := bc.candidate.Copy()
	cand.Algorithm = algo
	cand.Feedback = feedback
	cand.Fitness = math.MaxFloat64

	// Set the new operator's params
	cand.Ops[newOpIdx] = params

	// Mute all inactive operators
	allOps := make(map[int]bool)
	for _, op := range bc.activeOps {
		allOps[op] = true
	}
	allOps[newOpIdx] = true
	for opIdx := 0; opIdx < 6; opIdx++ {
		if !allOps[opIdx] {
			MuteOperator(cand, opIdx)
		}
	}

	return *cand
}

// evaluateSingleCandidate renders one candidate and returns its spectral distance.
func evaluateSingleCandidate(c *Candidate, target []float64, targetRMS float64, note uint8, dur time.Duration) float64 {
	audio := RenderCandidate(c, note, dur)
	trimmed := TrimOrPad(audio, len(target))
	rawRMS := CalcRMS(trimmed)
	normed := NormalizeRMS(trimmed)
	var dist float64
	if cachedTargetMel != nil {
		dist = MelCosineDistanceCached(cachedTargetMel, normed, RenderSampleRate)
	} else {
		dist = MelCosineDistance(target, normed, RenderSampleRate)
	}
	if targetRMS > 1e-4 {
		dist += CalcDBPenalty(rawRMS, targetRMS)
	}
	return dist
}

// evaluateBatch renders a slice of candidates in parallel and returns distances.
func evaluateBatch(candidates []*Candidate, target []float64, targetRMS float64, note uint8, dur time.Duration) []float64 {
	n := len(candidates)
	if n == 0 {
		return nil
	}

	// Use the existing parallel infrastructure
	dists, _ := evaluateBatchParallel(candidates, target, note, dur, targetRMS)
	return dists
}

// RunConstructiveBeam builds a DX7 patch incrementally, one operator at a time,
// recording the full trace of how each operator was added.
// If cb is non-nil, uses codebook entries for diverse exploration after each step.
// targetPeak and targetCachedSTFT enable the 3-tier peak-window filtering pipeline.
// timbreCB enables Tier 1 metadata pre-filtering (zero-render).
func RunConstructiveBeam(
	ctx context.Context,
	target []float64,
	targetRMS float64,
	note uint8,
	dur time.Duration,
	cache *OperatorCache,
	cb *Codebook,
	timbreCB *TimbreCodebook,
	targetPeak PeakShape,
	targetCachedSTFT *PrecomputedTargetSTFT,
	beamWidth int,
	maxRenders int,
	logFn func(string),
) *PatchEvolution {
	if beamWidth < 1 {
		beamWidth = 4
	}
	if logFn == nil {
		logFn = func(string) {}
	}

	var renderCount int
	var trace []OperatorAddition

	// ── Step 1: Single carrier (Op6, index 5) in Algorithm 31 (Algo 32) ──

	logFn("[Beam] Step 1: Finding best single carrier (Op6)...")

	step1Candidates := make([]*Candidate, 0, 3200)

	// Seed from cache or defaults
	feedbackLevels := []uint8{0, 3, 5, 7}
	freqCoarses := []uint8{1, 2, 3, 4, 5, 6, 7, 8}

	for _, fc := range freqCoarses {
		seeds := cache.GetSeeds("carrier", fc, 2)
		for _, seed := range seeds {
			for _, fb := range feedbackLevels {
				// Grid around seed
				for olDelta := -4; olDelta <= 4; olDelta += 2 {
					for l3Delta := -5; l3Delta <= 5; l3Delta += 5 {
						for r2Delta := -5; r2Delta <= 5; r2Delta += 5 {
							cand := InitVoice()
							cand.Algorithm = 31 // Algo 32
							cand.Feedback = fb
							cand.Ops[5] = OpParams{
								Mode:        0,
								FreqCoarse:  clampU8(seed.FreqCoarse, int8(fc-seed.FreqCoarse), 1, 31),
								FreqFine:    seed.FreqFine,
								OutputLevel: clampU8(seed.OutputLevel, int8(olDelta), 80, 99),
								Rate1:       99,
								Rate2:       clampU8(seed.Rate2, int8(r2Delta), 20, 99),
								Level3:      clampU8(seed.Level3, int8(l3Delta), 30, 99),
								Rate4:       40,
							}
							step1Candidates = append(step1Candidates, cand)
						}
					}
				}
			}
		}
	}

	// Evaluate step 1 candidates
	if len(step1Candidates) > maxRenders {
		step1Candidates = step1Candidates[:maxRenders]
	}
	dists := evaluateBatch(step1Candidates, target, targetRMS, note, dur)
	renderCount += len(step1Candidates)

	// Sort by distance, keep top beamWidth
	type scored struct {
		cand *Candidate
		dist float64
	}
	scored1 := make([]scored, len(step1Candidates))
	for i := range step1Candidates {
		scored1[i] = scored{step1Candidates[i], dists[i]}
	}
	sort.Slice(scored1, func(i, j int) bool { return scored1[i].dist < scored1[j].dist })

	beam := make([]beamCandidate, beamWidth)
	for i := 0; i < beamWidth && i < len(scored1); i++ {
		beam[i] = beamCandidate{
			candidate:  *scored1[i].cand,
			activeOps:  []int{5},
			roles:      map[int]string{5: "carrier"},
			modTargets: map[int]int{},
			distance:   scored1[i].dist,
		}
	}

	bestDist := beam[0].distance
	fb0 := beam[0].candidate.Feedback
	trace = append(trace, OperatorAddition{
		Step:        0,
		OpIndex:     5,
		Role:        "carrier",
		Algorithm:   31,
		ModTarget:   -1,
		Params:      beam[0].candidate.Ops[5],
		Feedback:    fb0,
		Distance:    bestDist,
		Description: fmt.Sprintf("Op6 as carrier (Algo 32), FreqCoarse=%d, Feedback=%d", beam[0].candidate.Ops[5].FreqCoarse, fb0),
	})

	logFn(fmt.Sprintf("[Beam] Step 1: %s — dist=%.4f", trace[0].Description, bestDist))

	// ── Steps 2-6: Add operators in reverse index order ──

beamLoop:
	for step := 1; step < 6; step++ {
		// Check context cancellation (budget)
		select {
		case <-ctx.Done():
			logFn("[Beam] CANCELLED — stopping early")
			break beamLoop
		default:
		}

		newOpIdx := placementOrder[step]
		logFn(fmt.Sprintf("[Beam] Step %d: Adding Op%d (index %d)...", step+1, 6-newOpIdx, newOpIdx))

		type candidateScore struct {
			bc   beamCandidate
			dist float64
		}
		var allCandidates []candidateScore

		for _, bc := range beam {
			// Find valid algorithm/role combinations for this new operator
			roles := getAlgorithmsForConfig(bc.roles, newOpIdx)
			if len(roles) == 0 {
				continue
			}

			// For each valid combination, grid search parameters
			for _, ar := range roles {
				role := ar.role
				fc := uint8(1)
				if role == "modulator" {
					fc = uint8(2) // modulators often benefit from higher ratios
				}

				seeds := cache.GetSeeds(role, fc, 3)
				for _, seed := range seeds {
					freqCoarses := []uint8{1, 2, 3, 4, 5, 6, 8, 10, 12}
					for _, fco := range freqCoarses {
						for olLo, olHi := 40, 99; olLo <= olHi; olLo += 10 {
							for l3Lo, l3Hi := 30, 99; l3Lo <= l3Hi; l3Lo += 15 {
								params := OpParams{
									Mode:        0,
									FreqCoarse:  fco,
									FreqFine:    seed.FreqFine,
									OutputLevel: uint8((olLo + olHi) / 2),
									Rate1:       99,
									Rate2:       seed.Rate2,
									Level3:      uint8((l3Lo + l3Hi) / 2),
									Rate4:       40,
								}
								cand := prepareCandidate(&bc, ar.algo, newOpIdx, params, bc.candidate.Feedback)
								allCandidates = append(allCandidates, candidateScore{
									bc: beamCandidate{
										candidate:  cand,
										activeOps:  append([]int{}, bc.activeOps...),
										roles:      copyRoles(bc.roles),
										modTargets: copyTargets(bc.modTargets),
										distance:   0,
									},
									dist: 0,
								})
							}
						}
					}
				}
			}
		}

		if len(allCandidates) == 0 {
			logFn(fmt.Sprintf("[Beam] Step %d: No valid configurations found, stopping.", step+1))
			break
		}

		// Enforce render budget
		remaining := maxRenders - renderCount
		if remaining <= 0 {
			logFn("[Beam] Render budget exhausted.")
			break
		}
		if len(allCandidates) > remaining {
			allCandidates = allCandidates[:remaining]
		}

		// Batch evaluate
		batch := make([]*Candidate, len(allCandidates))
		for i := range allCandidates {
			batch[i] = &allCandidates[i].bc.candidate
		}
		dists := evaluateBatch(batch, target, targetRMS, note, dur)
		renderCount += len(batch)

		// Assign distances back
		for i := range allCandidates {
			allCandidates[i].dist = dists[i]
			allCandidates[i].bc.distance = dists[i]
		}

		// ── Codebook exploration: diverse starting points from similar patches ──
		remainingRenders := maxRenders - renderCount
		if cb != nil && remainingRenders > 100 {
			logFn(fmt.Sprintf("[Beam] Step %d: Exploring codebook (%d candidates so far)...", step+1, len(allCandidates)))

			// Temporarily build beam from top greedy results for codebook matching
			sort.Slice(allCandidates, func(i, j int) bool { return allCandidates[i].dist < allCandidates[j].dist })
			tempBeam := make([]beamCandidate, beamWidth)
			for i := 0; i < beamWidth && i < len(allCandidates); i++ {
				tempBeam[i] = allCandidates[i].bc
				tempBeam[i].activeOps = append([]int{}, allCandidates[i].bc.activeOps...)
				tempBeam[i].activeOps = append(tempBeam[i].activeOps, newOpIdx)
				algo := allCandidates[i].bc.candidate.Algorithm
				if IsCarrier(algo, newOpIdx) {
					tempBeam[i].roles[newOpIdx] = "carrier"
					tempBeam[i].modTargets[newOpIdx] = -1
				} else {
					tempBeam[i].roles[newOpIdx] = "modulator"
					tempBeam[i].modTargets[newOpIdx] = findModTarget(&allCandidates[i].bc, newOpIdx, algo)
				}
			}

			cbCandidates := exploreCodebookForStep(tempBeam, cb, timbreCB, newOpIdx, target, targetRMS, targetPeak, targetCachedSTFT, note, dur, remainingRenders)
			renderCount += len(cbCandidates)

			// Merge codebook candidates into the pool
			for _, cbc := range cbCandidates {
				allCandidates = append(allCandidates, candidateScore{bc: cbc, dist: cbc.distance})
			}

			// Re-sort merged pool
			sort.Slice(allCandidates, func(i, j int) bool { return allCandidates[i].dist < allCandidates[j].dist })

			if len(cbCandidates) > 0 {
				logFn(fmt.Sprintf("[Beam] Step %d: Codebook exploration found %d candidates, best=%.4f",
					step+1, len(cbCandidates), cbCandidates[0].distance))
			}
		}

		// Sort and keep top beamWidth
		sort.Slice(allCandidates, func(i, j int) bool { return allCandidates[i].dist < allCandidates[j].dist })

		beam = make([]beamCandidate, beamWidth)
		for i := 0; i < beamWidth && i < len(allCandidates); i++ {
			bc := allCandidates[i].bc
			bc.activeOps = append([]int{}, bc.activeOps...)
			bc.activeOps = append(bc.activeOps, newOpIdx)
			// Determine the actual role from the algorithm routing
			algo := allCandidates[i].bc.candidate.Algorithm
			if IsCarrier(algo, newOpIdx) {
				bc.roles[newOpIdx] = "carrier"
				bc.modTargets[newOpIdx] = -1
			} else {
				bc.roles[newOpIdx] = "modulator"
				bc.modTargets[newOpIdx] = findModTarget(&allCandidates[i].bc, newOpIdx, algo)
			}
			beam[i] = bc
		}

		// Record trace for best in this step
		best := allCandidates[0]
		role := "carrier"
		modTgt := -1
		algo := best.bc.candidate.Algorithm
		if !IsCarrier(algo, newOpIdx) {
			role = "modulator"
			modTgt = findModTarget(&best.bc, newOpIdx, algo)
		}
		desc := fmt.Sprintf("Op%d as %s", 6-newOpIdx, role)
		if modTgt >= 0 {
			desc += fmt.Sprintf("→Op%d", 6-modTgt)
		}
		desc += fmt.Sprintf(" (Algo %d), FreqCoarse=%d", algo+1, best.bc.candidate.Ops[newOpIdx].FreqCoarse)

		trace = append(trace, OperatorAddition{
			Step:        step,
			OpIndex:     newOpIdx,
			Role:        role,
			Algorithm:   algo,
			ModTarget:   modTgt,
			Params:      best.bc.candidate.Ops[newOpIdx],
			Feedback:    best.bc.candidate.Feedback,
			Distance:    best.dist,
			Description: desc,
		})

		if best.dist < bestDist {
			bestDist = best.dist
		}
		logFn(fmt.Sprintf("[Beam] Step %d: %s — dist=%.4f (best=%.4f)", step+1, desc, best.dist, bestDist))
	}

	// Build final result
	finalCand := beam[0].candidate
	finalAlgo := finalCand.Algorithm

	logFn(fmt.Sprintf("[Beam] Complete: %d ops, final dist=%.4f, %d renders, algo=%d",
		len(beam[0].activeOps), beam[0].distance, renderCount, finalAlgo+1))

	return &PatchEvolution{
		Final:        finalCand,
		Trace:        trace,
		Algorithm:    finalAlgo,
		BeamWidth:    beamWidth,
		TotalRenders: renderCount,
	}
}

// findModTarget determines which existing operator the new modulator targets,
// based on the algorithm routing.
func findModTarget(bc *beamCandidate, newOpIdx int, algo uint8) int {
	newRoute := algoRoutingTable[algo][newOpIdx]
	for _, opIdx := range bc.activeOps {
		targetRoute := algoRoutingTable[algo][opIdx]
		if targetRoute.outBus == newRoute.inBus && newRoute.inBus != 0 {
			return opIdx
		}
	}
	return -1
}

// getRoleForOp returns the role of an operator in a beam candidate's configuration.
func getRoleForOp(bc *beamCandidate, opIdx int) string {
	if r, ok := bc.roles[opIdx]; ok {
		return r
	}
	return ""
}

func copyRoles(src map[int]string) map[int]string {
	dst := make(map[int]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func copyTargets(src map[int]int) map[int]int {
	dst := make(map[int]int, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// exploreCodebookForStep implements a 3-tier filtering pipeline for codebook entries:
//
//	Tier 1: Peak shape metadata filter (zero render, ~0.01ms per entry)
//	Tier 2: Short burst render + peak window spectral comparison (~300ms render)
//	Tier 3: Full evaluation only for top survivors
func exploreCodebookForStep(
	beam []beamCandidate,
	cb *Codebook,
	timbreCB *TimbreCodebook,
	newOpIdx int,
	target []float64,
	targetRMS float64,
	targetPeak PeakShape,
	targetCachedSTFT *PrecomputedTargetSTFT,
	note uint8,
	dur time.Duration,
	remainingRenders int,
) []beamCandidate {
	if remainingRenders <= 0 {
		return nil
	}

	// Collect algorithms currently in the beam
	beamAlgos := make(map[uint8]bool)
	for _, bc := range beam {
		beamAlgos[bc.candidate.Algorithm] = true
	}

	// ── Tier 1: Peak Shape Metadata Filter ──
	// Use timbre codebook's precomputed composite peak shapes to rank entries
	// by envelope similarity. Zero render cost.
	type tier1Candidate struct {
		entry     TimbreCodebookEntry
		peakDist  float64
		beamMatch int // index of matching beam candidate
	}
	var tier1 []tier1Candidate

	if timbreCB != nil && len(timbreCB.Entries) > 0 {
		// Pre-filter: collect entries matching beam algorithms
		matching := make([]TimbreCodebookEntry, 0, 256)
		for _, entry := range timbreCB.Entries {
			if beamAlgos[entry.Candidate.Algorithm] {
				matching = append(matching, entry)
			}
		}

		// Rank by peak shape distance
		for _, entry := range matching {
			pd := PeakShapeDistance(targetPeak, entry.CompositePeak)
			// Find matching beam candidate index
			beamIdx := -1
			for bi, bc := range beam {
				if bc.candidate.Algorithm == entry.Candidate.Algorithm {
					beamIdx = bi
					break
				}
			}
			tier1 = append(tier1, tier1Candidate{
				entry:     entry,
				peakDist:  pd,
				beamMatch: beamIdx,
			})
		}

		// Sort by peak shape distance
		sort.Slice(tier1, func(i, j int) bool { return tier1[i].peakDist < tier1[j].peakDist })

		// Keep top 30% or 300 entries, whichever is smaller
		tier1Cap := len(tier1) * 30 / 100
		if tier1Cap < 300 {
			tier1Cap = 300
		}
		if tier1Cap > len(tier1) {
			tier1Cap = len(tier1)
		}
		tier1 = tier1[:tier1Cap]
	} else if cb != nil && len(cb.Entries) > 0 {
		// Fallback: use fingerprint codebook if no timbre codebook available
		for _, entry := range cb.Entries {
			if !beamAlgos[entry.Candidate.Algorithm] {
				continue
			}
			beamIdx := -1
			for bi, bc := range beam {
				if bc.candidate.Algorithm == entry.Candidate.Algorithm {
					beamIdx = bi
					break
				}
			}
			tier1 = append(tier1, tier1Candidate{
				entry: TimbreCodebookEntry{
					Name:      entry.Name,
					Candidate: entry.Candidate,
				},
				peakDist:  0,
				beamMatch: beamIdx,
			})
		}
	}

	if len(tier1) == 0 {
		return nil
	}

	// ── Tier 2: Short Burst Render + Peak Window Comparison ──
	// Render 300ms for each Tier 1 survivor, find energy peak, extract 50ms window,
	// compare to target's 50ms peak window via SpectralDistance.
	tier2RenderDur := 300 * time.Millisecond
	peakWindowMs := 50
	peakWindowSamples := RenderSampleRate * peakWindowMs / 1000

	// Extract target's peak window once
	var targetPeakWindow []float64
	if targetCachedSTFT != nil && len(target) > 0 {
		_, peakSample := ExtractTimbreWindowWithPeak(target, RenderSampleRate)
		// Extract 50ms centered on peak
		halfWin := peakWindowSamples / 2
		winStart := peakSample - halfWin
		if winStart < 0 {
			winStart = 0
		}
		winEnd := winStart + peakWindowSamples
		if winEnd > len(target) {
			winEnd = len(target)
			winStart = winEnd - peakWindowSamples
			if winStart < 0 {
				winStart = 0
			}
		}
		targetPeakWindow = target[winStart:winEnd]
	}

	type tier2Candidate struct {
		t1       tier1Candidate
		bc       beamCandidate
		peakDist float64
	}
	var tier2 []tier2Candidate

	// Batch render short bursts for Tier 1 survivors
	tier2BatchSize := len(tier1)
	if tier2BatchSize > remainingRenders {
		tier2BatchSize = remainingRenders
	}
	// Cap Tier 2 renders at 300
	if tier2BatchSize > 300 {
		tier2BatchSize = 300
	}

	if targetPeakWindow != nil && len(targetPeakWindow) >= 1024 {
		// Build candidates for batch rendering
		tier2Cands := make([]*Candidate, tier2BatchSize)
		tier2BeamMatches := make([]tier1Candidate, tier2BatchSize)
		for i := 0; i < tier2BatchSize; i++ {
			t1 := tier1[i]
			cand := InitVoice()
			cand.Algorithm = t1.entry.Candidate.Algorithm
			cand.Feedback = t1.entry.Candidate.Feedback

			// Copy active operator params from codebook entry
			bc := beam[t1.beamMatch]
			for _, opIdx := range bc.activeOps {
				cand.Ops[opIdx] = t1.entry.Candidate.Ops[opIdx]
			}
			cand.Ops[newOpIdx] = t1.entry.Candidate.Ops[newOpIdx]

			// Mute inactive operators
			allOps := make(map[int]bool)
			for _, op := range bc.activeOps {
				allOps[op] = true
			}
			allOps[newOpIdx] = true
			for opIdx := 0; opIdx < 6; opIdx++ {
				if !allOps[opIdx] {
					MuteOperator(cand, opIdx)
				}
			}

			// Copy LFO/scaling from codebook RawBase
			if len(t1.entry.Candidate.RawBase) == 128 {
				cand.RawBase = make([]byte, 128)
				copy(cand.RawBase, t1.entry.Candidate.RawBase)
				cand.RawBase[110] = cand.Algorithm
			}

			tier2Cands[i] = cand
			tier2BeamMatches[i] = t1
		}

		// Render all Tier 2 candidates in parallel (short 300ms renders)
		tier2Renders := RenderCandidatesParallel(tier2Cands, note, tier2RenderDur)

		// Compare each rendered burst's peak window to target peak window
		for i, audio := range tier2Renders {
			if audio == nil || len(audio) < peakWindowSamples {
				continue
			}

			// Find energy peak in rendered audio
			_, peakSample := ExtractTimbreWindowWithPeak(audio, RenderSampleRate)

			// Extract 50ms window centered on peak
			halfWin := peakWindowSamples / 2
			winStart := peakSample - halfWin
			if winStart < 0 {
				winStart = 0
			}
			winEnd := winStart + peakWindowSamples
			if winEnd > len(audio) {
				winEnd = len(audio)
				winStart = winEnd - peakWindowSamples
				if winStart < 0 {
					winStart = 0
				}
			}
			peakWindow := audio[winStart:winEnd]
			if len(peakWindow) < 1024 {
				continue
			}

			// Compute spectral distance on peak windows
			pkDist := SpectralDistance(targetPeakWindow, peakWindow)

			// Build beam candidate
			t1 := tier2BeamMatches[i]
			bc := beam[t1.beamMatch]
			newCand := *tier2Cands[i]
			newRoles := copyRoles(bc.roles)
			newTargets := copyTargets(bc.modTargets)
			newOps := append([]int{}, bc.activeOps...)
			newOps = append(newOps, newOpIdx)

			if IsCarrier(newCand.Algorithm, newOpIdx) {
				newRoles[newOpIdx] = "carrier"
				newTargets[newOpIdx] = -1
			} else {
				newRoles[newOpIdx] = "modulator"
				newTargets[newOpIdx] = findModTarget(&bc, newOpIdx, newCand.Algorithm)
			}

			tier2 = append(tier2, tier2Candidate{
				t1: t1,
				bc: beamCandidate{
					candidate:  newCand,
					activeOps:  newOps,
					roles:      newRoles,
					modTargets: newTargets,
					distance:   pkDist,
				},
				peakDist: pkDist,
			})
		}

		// Sort Tier 2 by peak window distance
		sort.Slice(tier2, func(i, j int) bool { return tier2[i].peakDist < tier2[j].peakDist })

		// Keep top 20% or 8, whichever is larger
		tier2Cap := len(tier2) * 20 / 100
		if tier2Cap < 8 {
			tier2Cap = 8
		}
		if tier2Cap > len(tier2) {
			tier2Cap = len(tier2)
		}
		tier2 = tier2[:tier2Cap]
	}

	// ── Tier 3: Full Evaluation ──
	// Only fully render and evaluate the Tier 2 survivors
	if len(tier2) == 0 {
		// Fallback: evaluate Tier 1 candidates directly
		if len(tier1) == 0 {
			return nil
		}
		fallbackCap := len(tier1)
		if fallbackCap > remainingRenders {
			fallbackCap = remainingRenders
		}
		tier1 = tier1[:fallbackCap]

		batch := make([]*Candidate, len(tier1))
		for i, t1 := range tier1 {
			cand := InitVoice()
			cand.Algorithm = t1.entry.Candidate.Algorithm
			cand.Feedback = t1.entry.Candidate.Feedback
			bc := beam[t1.beamMatch]
			for _, opIdx := range bc.activeOps {
				cand.Ops[opIdx] = t1.entry.Candidate.Ops[opIdx]
			}
			cand.Ops[newOpIdx] = t1.entry.Candidate.Ops[newOpIdx]
			allOps := make(map[int]bool)
			for _, op := range bc.activeOps {
				allOps[op] = true
			}
			allOps[newOpIdx] = true
			for opIdx := 0; opIdx < 6; opIdx++ {
				if !allOps[opIdx] {
					MuteOperator(cand, opIdx)
				}
			}
			if len(t1.entry.Candidate.RawBase) == 128 {
				cand.RawBase = make([]byte, 128)
				copy(cand.RawBase, t1.entry.Candidate.RawBase)
				cand.RawBase[110] = cand.Algorithm
			}
			batch[i] = cand
		}

		dists := evaluateBatch(batch, target, targetRMS, note, dur)

		var results []beamCandidate
		for i, t1 := range tier1 {
			bc := beam[t1.beamMatch]
			newCand := *batch[i]
			newRoles := copyRoles(bc.roles)
			newTargets := copyTargets(bc.modTargets)
			newOps := append([]int{}, bc.activeOps...)
			newOps = append(newOps, newOpIdx)
			if IsCarrier(newCand.Algorithm, newOpIdx) {
				newRoles[newOpIdx] = "carrier"
				newTargets[newOpIdx] = -1
			} else {
				newRoles[newOpIdx] = "modulator"
				newTargets[newOpIdx] = findModTarget(&bc, newOpIdx, newCand.Algorithm)
			}
			results = append(results, beamCandidate{
				candidate:  newCand,
				activeOps:  newOps,
				roles:      newRoles,
				modTargets: newTargets,
				distance:   dists[i],
			})
		}
		sort.Slice(results, func(i, j int) bool { return results[i].distance < results[j].distance })
		return results
	}

	// Full evaluation of Tier 2 survivors
	fullBatch := make([]*Candidate, len(tier2))
	for i, t2 := range tier2 {
		fullBatch[i] = &t2.bc.candidate
	}
	fullDists := evaluateBatch(fullBatch, target, targetRMS, note, dur)

	var results []beamCandidate
	for i := range tier2 {
		tier2[i].bc.distance = fullDists[i]
		results = append(results, tier2[i].bc)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].distance < results[j].distance })
	return results
}
