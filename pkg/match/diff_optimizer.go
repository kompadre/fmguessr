package match

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// DiffOptimizer implements a Jacobian steering vector optimizer based on the
// differential feature-domain phase-cancellation manifold.
//
// Rather than stochastic search, DiffOptimizer computes the local N x 256 Jacobian sensitivity
// matrix J = dF/d\theta in the STFT log-frequency spectrum feature domain and solves
// regularized Gauss-Newton normal equations (J^T J + \lambda I) \Delta\theta = J^T E
// to steer parameter delta vectors directly toward the target audio.
type DiffOptimizer struct {
	targetNorm   []float64
	rawTargetRMS float64
	currentPatch *Candidate
	bestPatch    *Candidate
	bestDist     float64
	note         uint8
	dur          time.Duration
	maxIter      int
	verbose      bool
	sampleRate   int
	rng          *rand.Rand
}

func NewDiffOptimizer(target []float64, initial *Candidate, maxIter int, verbose bool, note uint8, dur time.Duration, rawTargetRMS ...float64) *DiffOptimizer {
	targetNorm := NormalizeRMS(target)
	tRMS := 0.0
	if len(rawTargetRMS) > 0 && rawTargetRMS[0] > 0 {
		tRMS = rawTargetRMS[0]
	} else {
		tRMS = CalcRMS(target)
	}

	startPatch := initial
	if startPatch == nil {
		startPatch = DecodeVoicePatch(epiano)
	}

	return &DiffOptimizer{
		targetNorm:   targetNorm,
		rawTargetRMS: tRMS,
		currentPatch: startPatch.Copy(),
		bestPatch:    startPatch.Copy(),
		bestDist:     math.MaxFloat64,
		note:         note,
		dur:          dur,
		maxIter:      maxIter,
		verbose:      verbose,
		sampleRate:   44100,
		rng:          rand.New(rand.NewSource(42)),
	}
}

func (opt *DiffOptimizer) BestDist() float64 {
	return opt.bestDist
}

func (opt *DiffOptimizer) BestPatch() *Candidate {
	return opt.bestPatch
}

// CrunchWavFile loads an arbitrary WAV file from disk and crunches it using DiffOptimizer.
func CrunchWavFile(wavPath string, initial *Candidate, maxIter int, verbose bool) (*Candidate, float64, error) {
	samples, sr, err := LoadWav(wavPath)
	if err != nil {
		return nil, math.MaxFloat64, fmt.Errorf("crunch wav load: %w", err)
	}

	renderDur := 1 * time.Second
	if len(samples) > 0 {
		durSec := float64(len(samples)) / float64(sr)
		if durSec > 3.0 {
			durSec = 3.0
		}
		if durSec >= 0.5 {
			renderDur = time.Duration(durSec * float64(time.Second))
		}
	}

	targetPcm := samples
	if len(targetPcm) > int(renderDur.Seconds()*float64(sr)) {
		targetPcm = targetPcm[:int(renderDur.Seconds()*float64(sr))]
	}

	optimizer := NewDiffOptimizer(targetPcm, initial, maxIter, verbose, 60, renderDur)
	best := optimizer.Run()
	return best, optimizer.BestDist(), nil
}

func buildTimbreKnobs() []KnobDef {
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
	return knobs
}

func buildTemporalKnobs() []KnobDef {
	knobs := make([]KnobDef, 0, 27)
	for op := 0; op < 6; op++ {
		opIdx := op
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Rate1", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].Rate1) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].Rate1 = uint8(v) },
			min:    0, max: 99,
		})
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Rate2", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].Rate2) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].Rate2 = uint8(v) },
			min:    0, max: 99,
		})
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Rate4", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].Rate4) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].Rate4 = uint8(v) },
			min:    0, max: 99,
		})
		knobs = append(knobs, KnobDef{
			name:   fmt.Sprintf("Op%d.Level3", opIdx+1),
			getter: func(c *Candidate) int { return int(c.Ops[opIdx].Level3) },
			setter: func(c *Candidate, v int) { c.Ops[opIdx].Level3 = uint8(v) },
			min:    0, max: 99,
		})
	}
	knobs = append(knobs, KnobDef{
		name:   "Pitch.Rate1",
		getter: func(c *Candidate) int { return int(c.PitchRate1) },
		setter: func(c *Candidate, v int) { c.PitchRate1 = uint8(v) },
		min:    0, max: 99,
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
	return knobs
}

// RunTopologySweep evaluates all 32 DX7 algorithm topologies in parallel and switches to the best matching topology.
func (opt *DiffOptimizer) RunTopologySweep() float64 {
	cands := make([]*Candidate, 32)
	for algo := uint8(0); algo < 32; algo++ {
		c := opt.currentPatch.Copy()
		c.Algorithm = algo
		cands[algo] = c
	}

	fits, _ := evaluateBatchParallel(cands, opt.targetNorm, opt.note, opt.dur, opt.rawTargetRMS)
	bestAlgo := opt.currentPatch.Algorithm
	for algo, fit := range fits {
		if fit < opt.bestDist {
			opt.bestDist = fit
			bestAlgo = uint8(algo)
			opt.bestPatch = cands[algo].Copy()
			opt.currentPatch = cands[algo].Copy()
		}
	}
	if opt.verbose {
		fmt.Printf("[DiffOpt-TopologySweep] Selected best Algo %d (Dist: %.6f)\n", bestAlgo+1, opt.bestDist)
	}
	return opt.bestDist
}

// RunPhase executes Jacobian steering and coupled sweeps on a specific knob subset.
func (opt *DiffOptimizer) RunPhase(knobs []KnobDef, phaseName string, maxIters int) float64 {
	numKnobs := len(knobs)
	if numKnobs == 0 {
		return opt.bestDist
	}

	targetSpecVec := ExtractTemporalTimbreSpec(opt.targetNorm, opt.sampleRate).ToExtendedVector()
	numFeatures := len(targetSpecVec)

	for iter := 0; iter < maxIters; iter++ {
		improved := false

		currPcm := RenderCandidate(opt.currentPatch, opt.note, opt.dur)
		currNorm := NormalizeRMS(currPcm)
		currSpecVec := ExtractTemporalTimbreSpec(currNorm, opt.sampleRate).ToExtendedVector()

		// 1. Calculate Feature Residual E = F_Target - F_Current
		E := make([]float64, numFeatures)
		for b := 0; b < numFeatures; b++ {
			E[b] = targetSpecVec[b] - currSpecVec[b]
		}

		// 2. Build N x numFeatures Jacobian Sensitivity Matrix J
		J, deltaSteps := opt.buildJacobian(opt.currentPatch, knobs, currNorm, currSpecVec)

		// 3. Compute Gram Matrix A = J^T J (N x N) and b = J^T E (N)
		Gram := make([][]float64, numKnobs)
		for i := 0; i < numKnobs; i++ {
			Gram[i] = make([]float64, numKnobs)
		}
		bVec := make([]float64, numKnobs)

		for i := 0; i < numKnobs; i++ {
			for j := 0; j < numKnobs; j++ {
				sum := 0.0
				for b := 0; b < numFeatures; b++ {
					sum += J[i][b] * J[j][b]
				}
				Gram[i][j] = sum
			}
			sumE := 0.0
			for b := 0; b < numFeatures; b++ {
				sumE += J[i][b] * E[b]
			}
			bVec[i] = sumE
		}

		// 4. Multi-Damping Line Search Candidates
		dampings := []float64{1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0}
		scales := []float64{0.25, 0.5, 0.75, 1.0, 1.5, 2.0}

		var lineCands []*Candidate
		var lineLabels []string

		for _, lambda := range dampings {
			dampedGram := make([][]float64, numKnobs)
			for i := 0; i < numKnobs; i++ {
				dampedGram[i] = make([]float64, numKnobs)
				copy(dampedGram[i], Gram[i])
				dampedGram[i][i] += lambda
			}

			deltaTheta, ok := solveLinearSystem(numKnobs, dampedGram, bVec)
			if !ok {
				continue
			}

			for _, s := range scales {
				c := opt.currentPatch.Copy()
				var shifts []string
				for i, k := range knobs {
					dTheta := deltaTheta[i] * s
					if math.IsNaN(dTheta) || math.IsInf(dTheta, 0) {
						continue
					}

					knobStep := int(math.Round(dTheta * float64(deltaSteps[i])))
					if knobStep == 0 {
						continue
					}

					curVal := k.getter(c)
					newVal := curVal + knobStep
					if newVal < k.min {
						newVal = k.min
					}
					if newVal > k.max {
						newVal = k.max
					}

					if newVal != curVal {
						k.setter(c, newVal)
						shifts = append(shifts, fmt.Sprintf("%s%+d", k.name, newVal-curVal))
					}
				}

				if len(shifts) > 0 {
					lineCands = append(lineCands, c)
					label := fmt.Sprintf("[%s-Jacobian] damp=%.1e,scale=%.2f: %s", phaseName, lambda, s, shifts)
					lineLabels = append(lineLabels, label)
				}
			}
		}

		if len(lineCands) > 0 {
			fits, _ := evaluateBatchParallel(lineCands, opt.targetNorm, opt.note, opt.dur, opt.rawTargetRMS)
			for i, fit := range fits {
				if fit < opt.bestDist {
					opt.bestDist = fit
					opt.bestPatch = lineCands[i].Copy()
					opt.currentPatch = lineCands[i].Copy()
					improved = true
					if opt.verbose {
						fmt.Printf("[DiffOpt-%s] Iter %d: New Best Dist %.6f via %s\n", phaseName, iter+1, opt.bestDist, lineLabels[i])
					}
				}
			}
		}

		// Fallback: Single-Knob Sweeps + Coupled Operator Sweeps
		if !improved {
			coupledCands, coupledMeta := generateCoupledOperatorCandidates(opt.currentPatch, knobs)
			if len(coupledCands) > 0 {
				fits, _ := evaluateBatchParallel(coupledCands, opt.targetNorm, opt.note, opt.dur, opt.rawTargetRMS)
				for i, fit := range fits {
					if fit < opt.bestDist {
						opt.bestDist = fit
						opt.bestPatch = coupledCands[i].Copy()
						opt.currentPatch = coupledCands[i].Copy()
						improved = true
						if opt.verbose {
							fmt.Printf("[DiffOpt-%s-Coupled] Iter %d: New Best Dist %.6f via %s\n", phaseName, iter+1, opt.bestDist, coupledMeta[i].knobName)
						}
					}
				}
			}
		}

		if !improved {
			break
		}
	}

	return opt.bestDist
}

// RunAlternating performs a 32-algorithm topology sweep followed by alternating
// back-to-back Timbre and Temporal optimization phases.
func (opt *DiffOptimizer) RunAlternating(cycles int) *Candidate {
	initialFits, _ := evaluateBatchParallel([]*Candidate{opt.currentPatch}, opt.targetNorm, opt.note, opt.dur, opt.rawTargetRMS)
	if len(initialFits) > 0 && initialFits[0] < opt.bestDist {
		opt.bestDist = initialFits[0]
	}

	if opt.verbose {
		fmt.Printf("[DiffOpt-Start] Initial Distance: %.6f\n", opt.bestDist)
	}

	if opt.bestDist <= 1e-4 {
		return opt.bestPatch
	}

	opt.RunTopologySweep()

	timbreKnobs := buildTimbreKnobs()
	temporalKnobs := buildTemporalKnobs()

	for c := 0; c < cycles; c++ {
		startDist := opt.bestDist

		// Pass 1: Timbre Phase (Output Levels, Fine, Coarse, Feedback)
		opt.RunPhase(timbreKnobs, fmt.Sprintf("Cycle%d-Timbre", c+1), 2)

		// Pass 2: Temporal Phase (Rates 1..4, Level3, Pitch Rates/Levels)
		opt.RunPhase(temporalKnobs, fmt.Sprintf("Cycle%d-Temporal", c+1), 2)

		if opt.verbose {
			fmt.Printf("[DiffOpt-Cycle %d] Dist: %.6f -> %.6f (Delta: %.6f)\n",
				c+1, startDist, opt.bestDist, startDist-opt.bestDist)
		}

		if startDist-opt.bestDist < 1e-4 {
			if opt.verbose {
				fmt.Printf("[DiffOpt-Alternating] Converged at cycle %d\n", c+1)
			}
			break
		}
	}

	return opt.bestPatch
}

// Run executes the alternating back-to-back optimization loop.
func (opt *DiffOptimizer) Run() *Candidate {
	return opt.RunAlternating(3)
}

// buildJacobian computes single-knob micro-shifted feature sensitivities for all knobs.
// J[i][b] is the sensitivity dF_b / d\theta_i for knob i and feature index b.
func (opt *DiffOptimizer) buildJacobian(base *Candidate, knobs []KnobDef, currNorm []float64, currSpecVec []float64) ([][]float64, []int) {
	numKnobs := len(knobs)
	numFeatures := len(currSpecVec)
	J := make([][]float64, numKnobs)
	for i := 0; i < numKnobs; i++ {
		J[i] = make([]float64, numFeatures)
	}
	deltaSteps := make([]int, numKnobs)

	cands := make([]*Candidate, numKnobs)
	for i, k := range knobs {
		c := base.Copy()
		curVal := k.getter(c)

		delta := 3
		if curVal+delta > k.max {
			delta = -3
		}
		if curVal+delta < k.min {
			delta = 1
		}

		k.setter(c, curVal+delta)
		cands[i] = c
		deltaSteps[i] = delta
	}

	// Render micro-shifted candidates in parallel
	fits, _ := evaluateBatchParallel(cands, opt.targetNorm, opt.note, opt.dur, opt.rawTargetRMS)
	_ = fits

	for i := 0; i < numKnobs; i++ {
		pcm := RenderCandidate(cands[i], opt.note, opt.dur)
		normPcm := NormalizeRMS(pcm)
		specVec := ExtractTemporalTimbreSpec(normPcm, opt.sampleRate).ToExtendedVector()

		step := float64(deltaSteps[i])
		for b := 0; b < numFeatures; b++ {
			J[i][b] = (specVec[b] - currSpecVec[b]) / step
		}
	}

	return J, deltaSteps
}

// solveLinearSystem solves Ax = b for an N x N system using Gaussian elimination with partial pivoting.
func solveLinearSystem(N int, A [][]float64, b []float64) ([]float64, bool) {
	M := make([][]float64, N)
	for i := 0; i < N; i++ {
		M[i] = make([]float64, N+1)
		for j := 0; j < N; j++ {
			M[i][j] = A[i][j]
		}
		M[i][N] = b[i]
	}

	// Forward elimination
	for k := 0; k < N; k++ {
		// Pivot selection
		maxRow := k
		maxVal := math.Abs(M[k][k])
		for i := k + 1; i < N; i++ {
			if abs := math.Abs(M[i][k]); abs > maxVal {
				maxVal = abs
				maxRow = i
			}
		}

		if maxVal < 1e-12 {
			return nil, false // Singular matrix
		}

		// Swap rows
		if maxRow != k {
			M[k], M[maxRow] = M[maxRow], M[k]
		}

		// Eliminate below
		pivot := M[k][k]
		for i := k + 1; i < N; i++ {
			factor := M[i][k] / pivot
			for j := k; j <= N; j++ {
				M[i][j] -= factor * M[k][j]
			}
		}
	}

	// Back substitution
	x := make([]float64, N)
	for i := N - 1; i >= 0; i-- {
		sum := M[i][N]
		for j := i + 1; j < N; j++ {
			sum -= M[i][j] * x[j]
		}
		x[i] = sum / M[i][i]
	}

	return x, true
}
