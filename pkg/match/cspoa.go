package match

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// CSPOAConfig configures the Codebook-Seeded Progressive Operator Attachment.
type CSPOAConfig struct {
	Enabled    bool
	MaxRenders int // Total render budget (default 5000)
	BeamWidth  int // Top-K candidates returned (default 3)
}

// DefaultCSPOAConfig returns sensible defaults.
func DefaultCSPOAConfig() CSPOAConfig {
	return CSPOAConfig{
		Enabled:    false,
		MaxRenders: 5000,
		BeamWidth:  3,
	}
}

// turnKind distinguishes mode transitions from knob adjustments.
type turnKind int

const (
	turnMode turnKind = iota
	turnKnob
)

// turn represents one available action: activating an operator mode or adjusting a knob.
type turn struct {
	kind      turnKind
	opIdx     int         // operator index (0-5)
	algo      uint8       // algorithm (mode turns only)
	role      string      // "carrier" or "modulator" (mode turns only)
	modTarget int         // modulator target op (mode turns only, -1 for carrier)
	knobName  string      // knob name (knob turns only)
	delta     int         // applied delta (knob turns only)
	cand      *Candidate  // resulting candidate if this turn is committed
	dist      float64     // spectral distance of cand
	pruned    bool        // true if this turn yielded zero improvement
}

// cspoaState tracks the full search state.
type cspoaState struct {
	candidate     *Candidate
	seedCandidate *Candidate // best codebook match for seeding mode turns
	activeOps     []int
	roles         map[int]string
	modTargets    map[int]int
	algorithm     uint8
	distance      float64

	knobProgress map[string]*knobProgress
	knobs        []KnobDef
}

type knobProgress struct {
	tested    map[int]bool // positions already evaluated
	bestDir   int          // +1 or -1, direction of improvement
	monotonic bool
	zeroCount int // how many deltas gave zero improvement
}

// RunCSPOA executes the Codebook-Seeded Progressive Operator Attachment.
// It returns the top-K candidates found, sorted by distance.
// seedCandidate is the best codebook match used to initialize operator parameters.
func RunCSPOA(
	target []float64,
	targetRMS float64,
	note uint8,
	dur time.Duration,
	deltaLib *DeltaLibrary,
	cache *OperatorCache,
	seedCandidate *Candidate,
	cfg CSPOAConfig,
	logFn func(string),
) []*Candidate {
	if cfg.MaxRenders <= 0 {
		cfg.MaxRenders = 5000
	}
	if cfg.BeamWidth <= 0 {
		cfg.BeamWidth = 3
	}
	if logFn == nil {
		logFn = func(string) {}
	}

	knobs := buildKnobs()

	// Initialize state from seed candidate or defaults
	state := &cspoaState{
		activeOps:     nil,
		roles:         make(map[int]string),
		modTargets:    make(map[int]int),
		knobProgress:  make(map[string]*knobProgress),
		knobs:         knobs,
		seedCandidate: seedCandidate,
	}

	if seedCandidate != nil {
		// Start from the codebook best candidate directly
		state.candidate = seedCandidate.Copy()
		state.algorithm = seedCandidate.Algorithm
		state.candidate.Fitness = math.MaxFloat64

		// Identify active operators from the seed's algorithm
		for i := 0; i < 6; i++ {
			if !isMuted(seedCandidate, i) {
				role := "carrier"
				if !IsCarrier(seedCandidate.Algorithm, i) {
					role = "modulator"
				}
				state.activeOps = append(state.activeOps, i)
				state.roles[i] = role
				if role == "modulator" {
					state.modTargets[i] = findModTargetForAlgo(seedCandidate.Algorithm, i)
				}
			}
		}
		logFn(fmt.Sprintf("[CSPOA] Seeded from codebook: Algo %d, %d active ops, Op6 Level=%d",
			state.algorithm+1, len(state.activeOps), state.candidate.Ops[5].OutputLevel))
	} else {
		// No codebook: start from InitVoice with dominant frequency
		dominantFreq := detectDominantFrequency(target)
		freqCoarseSeed := freqToCoarse(dominantFreq)
		logFn(fmt.Sprintf("[CSPOA] No seed candidate, dominant freq: %.1f Hz -> FreqCoarse: %d", dominantFreq, freqCoarseSeed))

		state.candidate = InitVoice()
		state.candidate.Algorithm = 31
		state.algorithm = 31
		state.candidate.Ops[5].FreqCoarse = uint8(freqCoarseSeed)
		state.candidate.Ops[5].OutputLevel = 85
		state.candidate.Ops[5].Level3 = 70
		state.candidate.Ops[5].Rate2 = 40
	}

	// Evaluate initial state
	state.distance = evalCandidate(state.candidate, target, targetRMS, note, dur)

	// All candidates pool for top-K tracking
	type scored struct {
		cand *Candidate
		dist float64
	}
	var allScored []scored
	allScored = append(allScored, scored{state.candidate.Copy(), state.distance})

	var renderCount int
	var totalTurns int

	logFn(fmt.Sprintf("[CSPOA] Initial state: dist=%.6f, algo=%d", state.distance, state.algorithm+1))

	// Main loop: iteratively pick the best available turn
	maxIterations := 200
	for iter := 0; iter < maxIterations && renderCount < cfg.MaxRenders; iter++ {
		// Generate all available turns
		turns := generateTurns(state, deltaLib)
		if len(turns) == 0 {
			logFn(fmt.Sprintf("[CSPOA] No available turns at iteration %d, stopping.", iter))
			break
		}

		// Filter out already-tested knob positions
		turns = filterTestedTurns(state, turns)
		if len(turns) == 0 {
			logFn(fmt.Sprintf("[CSPOA] All turns exhausted at iteration %d, stopping.", iter))
			break
		}

		// Batch evaluate all turns
		cands := make([]*Candidate, len(turns))
		for i, t := range turns {
			cands[i] = t.cand
		}

		remaining := cfg.MaxRenders - renderCount
		if len(cands) > remaining {
			cands = cands[:remaining]
			turns = turns[:remaining]
		}

		fits, _ := evaluateBatchParallel(cands, target, note, dur, targetRMS)
		renderCount += len(cands)

		// Assign distances back to turns
		for i, f := range fits {
			turns[i].dist = f
		}

		// Sort turns by distance (ascending = best first)
		sort.Slice(turns, func(i, j int) bool {
			return turns[i].dist < turns[j].dist
		})

		// Pick best improving turn
		bestTurn := turns[0]
		if bestTurn.dist >= state.distance {
			// No improvement found — mark all as pruned
			for _, t := range turns {
				if t.kind == turnKnob {
					key := fmt.Sprintf("op%d.%s", t.opIdx, t.knobName)
					if kp, ok := state.knobProgress[key]; ok {
						kp.zeroCount++
					}
				}
			}
			logFn(fmt.Sprintf("[CSPOA] Iter %d: No improving turn (best=%.6f >= current=%.6f), stopping.",
				iter, bestTurn.dist, state.distance))
			break
		}

		// Commit best turn
		prevDist := state.distance
		commitTurn(state, bestTurn)
		totalTurns++
		allScored = append(allScored, scored{state.candidate.Copy(), state.distance})

		if state.distance < allScored[0].dist {
			allScored[0] = scored{state.candidate.Copy(), state.distance}
		}

		if logFn != nil && (totalTurns <= 10 || totalTurns%5 == 0 || state.distance < prevDist*0.9) {
			desc := turnDescription(bestTurn)
			logFn(fmt.Sprintf("[CSPOA] Turn %d: %s (dist=%.6f -> %.6f, Δ=%.6f)",
				totalTurns, desc, prevDist, state.distance, state.distance-prevDist))
		}

		// Mark zero-impact knob turns as pruned
		for _, t := range turns[1:] {
			if t.kind == turnKnob && math.Abs(t.dist-state.distance) < 1e-6 {
				t.pruned = true
			}
		}
	}

	logFn(fmt.Sprintf("[CSPOA] Complete: %d turns, %d renders, best=%.6f",
		totalTurns, renderCount, state.distance))

	// Return top-K candidates
	sort.Slice(allScored, func(i, j int) bool {
		return allScored[i].dist < allScored[j].dist
	})

	k := cfg.BeamWidth
	if k > len(allScored) {
		k = len(allScored)
	}
	result := make([]*Candidate, k)
	for i := 0; i < k; i++ {
		result[i] = allScored[i].cand
		result[i].Fitness = allScored[i].dist
	}
	return result
}

// generateTurns builds all currently available turns from the state.
func generateTurns(state *cspoaState, deltaLib *DeltaLibrary) []*turn {
	var turns []*turn

	// Mode turns: for each muted operator, test all valid algorithm/role combos
	for opIdx := 0; opIdx < 6; opIdx++ {
		if _, active := state.roles[opIdx]; active {
			continue
		}

		validCombos := getAlgorithmsForConfig(state.roles, opIdx)
		for _, combo := range validCombos {
			cand := state.candidate.Copy()
			cand.Algorithm = combo.algo

			// Set new operator params from seed candidate if available
			role := combo.role
			fc := uint8(1)
			if role == "modulator" {
				fc = uint8(2)
			}
			var seedParams OpParams
			if state.seedCandidate != nil {
				seedParams = state.seedCandidate.Ops[opIdx]
				// Ensure modulator level isn't too high (reduces modulation index)
				if role == "modulator" && seedParams.OutputLevel > 80 {
					seedParams.OutputLevel = 65
				}
				// Ensure carrier level is reasonable
				if role == "carrier" && seedParams.OutputLevel < 70 {
					seedParams.OutputLevel = 85
				}
			} else {
				seedParams = defaultOpParams(role, fc)
			}
			cand.Ops[opIdx] = seedParams

			// Mute all still-inactive operators
			for i := 0; i < 6; i++ {
				if _, ok := state.roles[i]; !ok && i != opIdx {
					MuteOperator(cand, i)
				}
			}

			turns = append(turns, &turn{
				kind:      turnMode,
				opIdx:     opIdx,
				algo:      combo.algo,
				role:      combo.role,
				modTarget: combo.target,
				cand:      cand,
			})
		}
	}

	// Knob turns: for each active operator, test codebook deltas on each knob
	for _, opIdx := range state.activeOps {
		role := state.roles[opIdx]

		for _, knob := range state.knobs {
			// Only knobs belonging to this operator
			if !knobBelongsToOp(knob, opIdx) {
				continue
			}

			curVal := knob.getter(state.candidate)
			deltas := deltaLib.GetDeltas(role, knob.name)

			for _, d := range deltas {
				newVal := curVal + d.Delta
				if newVal < knob.min || newVal > knob.max {
					continue
				}
				if newVal == curVal {
					continue
				}

				cand := state.candidate.Copy()
				knob.setter(cand, newVal)

				turns = append(turns, &turn{
					kind:     turnKnob,
					opIdx:    opIdx,
					knobName: knob.name,
					delta:    d.Delta,
					cand:     cand,
				})
			}

			// Also test median position if available and different from current
			median := deltaLib.GetMedian(role, knob.name)
			if median != curVal && median >= knob.min && median <= knob.max {
				cand := state.candidate.Copy()
				knob.setter(cand, median)
				turns = append(turns, &turn{
					kind:     turnKnob,
					opIdx:    opIdx,
					knobName: knob.name,
					delta:    median - curVal,
					cand:     cand,
				})
			}
		}
	}

	// Feedback knob (global, applies to all active configs)
	if len(state.activeOps) > 0 {
		fbDeltas := deltaLib.GetDeltas("carrier", "Feedback")
		curFb := int(state.candidate.Feedback)
		for _, d := range fbDeltas {
			newFb := curFb + d.Delta
			if newFb < 0 || newFb > 7 {
				continue
			}
			if newFb == curFb {
				continue
			}
			cand := state.candidate.Copy()
			cand.Feedback = uint8(newFb)
			turns = append(turns, &turn{
				kind:     turnKnob,
				opIdx:    -1, // global
				knobName: "Feedback",
				delta:    d.Delta,
				cand:     cand,
			})
		}
	}

	return turns
}

// filterTestedTurns removes turns whose knob position was already tested.
func filterTestedTurns(state *cspoaState, turns []*turn) []*turn {
	var filtered []*turn
	for _, t := range turns {
		if t.kind == turnKnob {
			key := fmt.Sprintf("op%d.%s", t.opIdx, t.knobName)
			kp, ok := state.knobProgress[key]
			if ok {
				var newVal int
				if t.knobName == "Feedback" {
					newVal = int(t.cand.Feedback)
				} else {
					newVal = getKnobValue(t.cand, t.opIdx, t.knobName)
				}
				if kp.tested[newVal] {
					continue
				}
			}
		}
		filtered = append(filtered, t)
	}
	return filtered
}

// commitTurn applies a turn to the state and updates all tracking.
func commitTurn(state *cspoaState, t *turn) {
	state.candidate = t.cand.Copy()

	if t.kind == turnMode {
		// Activate new operator
		state.activeOps = append(state.activeOps, t.opIdx)
		state.roles[t.opIdx] = t.role
		state.modTargets[t.opIdx] = t.modTarget
		state.algorithm = t.algo
		state.candidate.Algorithm = t.algo

		// Initialize knob progress for newly activated operator
		for _, knob := range state.knobs {
			if knobBelongsToOp(knob, t.opIdx) {
				key := fmt.Sprintf("op%d.%s", t.opIdx, knob.name)
				val := knob.getter(state.candidate)
				state.knobProgress[key] = &knobProgress{
					tested:    map[int]bool{val: true},
					monotonic: true,
				}
			}
		}
	} else {
		// Knob turn: record tested position
		key := fmt.Sprintf("op%d.%s", t.opIdx, t.knobName)
		kp, ok := state.knobProgress[key]
		if !ok {
			kp = &knobProgress{
				tested:    make(map[int]bool),
				monotonic: true,
			}
			state.knobProgress[key] = kp
		}

		newVal := getKnobValue(state.candidate, t.opIdx, t.knobName)
		kp.tested[newVal] = true

		// Update monotonicity tracking
		updateMonotonicity(kp, t)
	}

	state.distance = t.dist
}

// updateMonotonicity checks if the knob's gradient direction is consistent.
func updateMonotonicity(kp *knobProgress, t *turn) {
	// Simple heuristic: if this turn improved distance, it's going in the right direction
	if t.dist < 1e-6 {
		return
	}
	// We track direction through the delta sign and whether it improved
	if kp.bestDir == 0 {
		if t.delta > 0 {
			kp.bestDir = 1
		} else {
			kp.bestDir = -1
		}
	}
}

// knobBelongsToOp checks if a knob definition belongs to a given operator index.
func knobBelongsToOp(knob KnobDef, opIdx int) bool {
	prefix := fmt.Sprintf("Op%d.", opIdx+1)
	return len(knob.name) > len(prefix) && knob.name[:len(prefix)] == prefix
}

// getKnobValue reads a knob's current value from a candidate.
func getKnobValue(c *Candidate, opIdx int, knobName string) int {
	for _, k := range buildKnobs() {
		if k.name == knobName {
			return k.getter(c)
		}
	}
	return 0
}

// defaultOpParams returns sensible default parameters for a new operator.
func defaultOpParams(role string, freqCoarse uint8) OpParams {
	if role == "carrier" {
		return OpParams{
			Mode:        0,
			FreqCoarse:  freqCoarse,
			FreqFine:    0,
			OutputLevel: 85,
			Rate1:       99,
			Rate2:       40,
			Level3:      70,
			Rate4:       40,
		}
	}
	return OpParams{
		Mode:        0,
		FreqCoarse:  freqCoarse,
		FreqFine:    0,
		OutputLevel: 65,
		Rate1:       99,
		Rate2:       35,
		Level3:      60,
		Rate4:       40,
	}
}

// evalCandidate renders a candidate and returns its spectral distance.
func evalCandidate(c *Candidate, target []float64, targetRMS float64, note uint8, dur time.Duration) float64 {
	audio := RenderCandidate(c, note, dur)
	trimmed := TrimOrPad(audio, len(target))
	rawRMS := CalcRMS(trimmed)
	normed := NormalizeRMS(trimmed)
	dist := SpectralDistance(target, normed)
	if targetRMS > 1e-4 {
		dist += CalcDBPenalty(rawRMS, targetRMS)
	}
	return dist
}

// turnDescription returns a human-readable description of a turn.
func turnDescription(t *turn) string {
	if t.kind == turnMode {
		role := t.role
		target := ""
		if t.modTarget >= 0 {
			target = fmt.Sprintf("->Op%d", t.modTarget+1)
		}
		return fmt.Sprintf("Mode Op%d as %s%s (Algo %d)", t.opIdx+1, role, target, t.algo+1)
	}
	return fmt.Sprintf("Op%d.%s%+d", t.opIdx+1, t.knobName, t.delta)
}

// detectDominantFrequency finds the strongest frequency component in the target audio.
func detectDominantFrequency(audio []float64) float64 {
	if len(audio) == 0 {
		return 440.0
	}

	// Simple autocorrelation-based pitch detection
	n := len(audio)
	minPeriod := 44   // 1000 Hz max
	maxPeriod := 882  // 50 Hz min

	if maxPeriod >= n {
		maxPeriod = n - 1
	}
	if minPeriod < 1 {
		minPeriod = 1
	}

	bestCorr := 0.0
	bestPeriod := minPeriod

	for period := minPeriod; period <= maxPeriod && period < n; period++ {
		corr := 0.0
		energy := 0.0
		for i := 0; i < n-period; i++ {
			corr += audio[i] * audio[i+period]
			energy += audio[i] * audio[i]
		}
		if energy > 0 {
			corr /= energy
		}
		if corr > bestCorr {
			bestCorr = corr
			bestPeriod = period
		}
	}

	if bestPeriod > 0 {
		return 44100.0 / float64(bestPeriod)
	}
	return 440.0
}

// freqToCoarse maps a frequency in Hz to the nearest DX7 FreqCoarse value.
// DX7 frequency ratio = FreqCoarse * (1 + FreqFine/100) for ratio mode.
// Coarse values 1-31 map to ratios 1.00 to 31.00.
func freqToCoarse(freq float64) int {
	if freq <= 0 {
		return 1
	}
	// Assume fundamental is the detected frequency
	// Map to coarse ratio relative to MIDI note C4 (261.63 Hz)
	coarse := int(math.Round(freq / 261.63))
	if coarse < 1 {
		coarse = 1
	}
	if coarse > 31 {
		coarse = 31
	}
	return coarse
}

// isMuted checks if an operator is muted (OutputLevel == 0 and Level3 == 0).
func isMuted(c *Candidate, opIdx int) bool {
	return c.Ops[opIdx].OutputLevel == 0 && c.Ops[opIdx].Level3 == 0
}

// findModTargetForAlgo finds which operator a modulator targets in a given algorithm.
func findModTargetForAlgo(algo uint8, opIdx int) int {
	routing := algoRoutingTable[algo][opIdx]
	if routing.isCarrier {
		return -1
	}
	// Find the operator whose output bus matches this operator's input bus
	for targetOp := 0; targetOp < 6; targetOp++ {
		if targetOp == opIdx {
			continue
		}
		targetRouting := algoRoutingTable[algo][targetOp]
		if targetRouting.outBus == routing.inBus {
			return targetOp
		}
	}
	return -1
}
