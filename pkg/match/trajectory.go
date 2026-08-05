package match

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"
)

// TrajectorySpec holds sliding-window embeddings for a PCM signal.
type TrajectorySpec struct {
	Embeddings [][]float32 // [TimeSteps][EmbDim]
	TimeSteps  int
	EmbDim     int
}

// PhaseBounds holds auto-detected envelope phase boundaries.
type PhaseBounds struct {
	AttackEnd  int // time step index where attack → sustain
	SustainEnd int // time step index where sustain → release
}

// TrajectoryConfig controls trajectory distance computation.
type TrajectoryConfig struct {
	WeightAttack  float64 // weight for attack phase distance (default 2.0)
	WeightSustain float64 // weight for sustain phase distance (default 1.0)
	WeightRelease float64 // weight for release phase distance (default 0.8)
	LFOLambda     float64 // weight for LFO spectrum penalty after Z-score norm (default 0.20)
	LFOBins       int     // number of FFT bins for LFO spectrum (default 64)
	LFONormAlpha  float64 // EMA smoothing for Z-score stats (default 0.01)
}

// DefaultTrajectoryConfig returns sensible defaults.
func DefaultTrajectoryConfig() TrajectoryConfig {
	return TrajectoryConfig{
		WeightAttack:  2.0,
		WeightSustain: 1.0,
		WeightRelease: 0.8,
		LFOLambda:     0.20,
		LFOBins:       64,
		LFONormAlpha:  0.01,
	}
}

// LFONormStats tracks running EMA for Z-score normalization of D_LFO.
type LFONormStats struct {
	Mean  float64
	Var   float64
	Alpha float64
}

// NewLFONormStats creates stats with the given EMA smoothing factor.
func NewLFONormStats(alpha float64) *LFONormStats {
	return &LFONormStats{Alpha: alpha}
}

// Update incorporates a new D_LFO sample and returns the Z-score normalized value.
func (s *LFONormStats) Update(raw float64) float64 {
	if s.Alpha == 0 {
		s.Alpha = 0.01
	}
	s.Mean += s.Alpha * (raw - s.Mean)
	s.Var += s.Alpha * ((raw-s.Mean)*(raw-s.Mean) - s.Var)
	std := math.Sqrt(s.Var)
	if std < 1e-8 {
		return 0
	}
	return (raw - s.Mean) / std
}

// ExtractTrajectory computes sliding-window embeddings from PCM audio.
// Returns one embedding per 30ms step, each computed from 250ms context.
func ExtractTrajectory(enc *AudioEncoder, samples []float64) *TrajectorySpec {
	if enc == nil || len(samples) == 0 {
		return nil
	}
	embeddings := enc.EmbedSequence(samples)
	if len(embeddings) == 0 {
		return nil
	}
	return &TrajectorySpec{
		Embeddings: embeddings,
		TimeSteps:  len(embeddings),
		EmbDim:     len(embeddings[0]),
	}
}

// DetectPhaseBounds auto-detects attack/sustain/release boundaries from
// the target audio's broadband RMS energy envelope.
//
// AttackEnd: time step where energy drops to 80% of peak (or first inflection)
// SustainEnd: time step where energy drops below -24dB relative to sustain floor
func DetectPhaseBounds(samples []float64, sampleRate int) PhaseBounds {
	if len(samples) == 0 || sampleRate == 0 {
		return PhaseBounds{AttackEnd: 5, SustainEnd: 50}
	}

	// Compute frame-level RMS energy (25ms frames, matching trajectory stride)
	frameSize := sampleRate / 40 // 25ms
	if frameSize < 1 {
		frameSize = 1
	}
	nFrames := len(samples) / frameSize
	if nFrames < 3 {
		return PhaseBounds{AttackEnd: 1, SustainEnd: nFrames}
	}

	energy := make([]float64, nFrames)
	peakEnergy := 0.0
	peakIdx := 0
	for i := 0; i < nFrames; i++ {
		start := i * frameSize
		end := start + frameSize
		if end > len(samples) {
			end = len(samples)
		}
		sum := 0.0
		for _, s := range samples[start:end] {
			sum += s * s
		}
		rms := math.Sqrt(sum / float64(end-start))
		energy[i] = rms
		if rms > peakEnergy {
			peakEnergy = rms
			peakIdx = i
		}
	}

	// AttackEnd: first frame after peak where energy drops below 80% of peak
	attackEnd := peakIdx
	attackThreshold := peakEnergy * 0.8
	for i := peakIdx; i < nFrames; i++ {
		if energy[i] < attackThreshold {
			attackEnd = i
			break
		}
		attackEnd = i
	}
	if attackEnd < 1 {
		attackEnd = 1
	}

	// SustainEnd: find where energy drops below -24dB relative to peak
	// -24dB = peak * 10^(-24/20) ≈ peak * 0.063
	sustainThreshold := peakEnergy * 0.063
	sustainEnd := nFrames - 1
	// Search backward from end to find release start
	for i := nFrames - 1; i > attackEnd; i-- {
		if energy[i] > sustainThreshold {
			sustainEnd = i + 1
			break
		}
	}
	if sustainEnd <= attackEnd {
		sustainEnd = attackEnd + 1
	}
	if sustainEnd >= nFrames {
		sustainEnd = nFrames - 1
	}

	// Convert frame indices to trajectory time step indices
	// Trajectory uses 30ms stride; we used 25ms frames above — close enough
	return PhaseBounds{
		AttackEnd:  attackEnd,
		SustainEnd: sustainEnd,
	}
}

// ComputeLFOSpectrum computes the FFT magnitude spectrum of a template
// distance trajectory for LFO penalty comparison.
// Input: a trajectory spec (typically the target's).
// Output: magnitude spectrum of the first LFOBins bins of |FFT(d(t))|.
func ComputeLFOSpectrum(traj *TrajectorySpec) []float64 {
	if traj == nil || traj.TimeSteps < 4 {
		return nil
	}
	// Use self-distance (all zeros) as template — the LFO pattern comes from
	// the temporal structure of the embeddings themselves.
	// We compute the spectral energy of each embedding dimension over time,
	// then average to get a "modulation spectrum".
	n := nextPowerOf2(traj.TimeSteps)
	lfoBins := 64
	if lfoBins > n/2 {
		lfoBins = n / 2
	}

	spectrum := make([]float64, lfoBins)
	dim := traj.EmbDim

	for d := 0; d < dim; d++ {
		// Extract this dimension's time series
		signal := make([]float64, n)
		for t := 0; t < traj.TimeSteps; t++ {
			signal[t] = float64(traj.Embeddings[t][d])
		}
		// Simple DFT magnitude for first lfoBins frequencies
		for k := 0; k < lfoBins; k++ {
			re, im := 0.0, 0.0
			for t := 0; t < n; t++ {
				angle := -2.0 * math.Pi * float64(k*t) / float64(n)
				re += signal[t] * math.Cos(angle)
				im += signal[t] * math.Sin(angle)
			}
			spectrum[k] += math.Sqrt(re*re + im*im)
		}
	}

	// Normalize by dimension count
	for i := range spectrum {
		spectrum[i] /= float64(dim)
	}
	return spectrum
}

// TrajectoryDistance computes the hybrid weighted distance between two trajectories.
// Uses phased weighting (attack/sustain/release) + Z-score normalized LFO penalty.
func TrajectoryDistance(target, cand *TrajectorySpec, cfg TrajectoryConfig,
	targetLFO []float64, lfoStats *LFONormStats) float64 {

	if target == nil || cand == nil {
		return math.MaxFloat64
	}

	n := target.TimeSteps
	if cand.TimeSteps < n {
		n = cand.TimeSteps
	}
	if n == 0 {
		return math.MaxFloat64
	}

	// Per-frame cosine distances
	d := make([]float64, n)
	for t := 0; t < n; t++ {
		d[t] = cosineDistFloat32(target.Embeddings[t], cand.Embeddings[t])
	}

	// Phase boundaries (use target's bounds)
	attackEnd := cfg.LFOBins // fallback
	sustainEnd := n
	// Phase bounds are passed via the trajectory config or detected externally.
	// For now, use simple heuristic: attack = first 10%, sustain = 10%-80%, release = last 20%
	attackEnd = n / 10
	if attackEnd < 1 {
		attackEnd = 1
	}
	sustainEnd = n * 4 / 5
	if sustainEnd <= attackEnd {
		sustainEnd = attackEnd + 1
	}

	// Phased distances
	D_attack := 0.0
	for t := 0; t < attackEnd && t < n; t++ {
		if d[t] > D_attack {
			D_attack = d[t] // max for attack
		}
	}

	D_sustain := 0.0
	sustainCount := 0
	for t := attackEnd; t < sustainEnd && t < n; t++ {
		D_sustain += d[t]
		sustainCount++
	}
	if sustainCount > 0 {
		D_sustain /= float64(sustainCount)
	}

	D_release := 0.0
	releaseCount := 0
	for t := sustainEnd; t < n; t++ {
		D_release += d[t]
		releaseCount++
	}
	if releaseCount > 0 {
		D_release /= float64(releaseCount)
	}

	D_phased := cfg.WeightAttack*D_attack + cfg.WeightSustain*D_sustain + cfg.WeightRelease*D_release

	// LFO penalty
	if len(targetLFO) > 0 && lfoStats != nil && cfg.LFOBins > 0 {
		// Compute candidate LFO spectrum from its distance trajectory
		candLFO := distanceTrajectorySpectrum(d, cfg.LFOBins)
		D_LFO_raw := cosineDistFloat64(targetLFO, candLFO)
		D_LFO_norm := lfoStats.Update(D_LFO_raw)
		D_phased += cfg.LFOLambda * D_LFO_norm
	}

	return D_phased
}

// TrajectoryGradient computes the scalar gradient ∂d(t*)/∂θ_i at the worst time step.
// Returns one gradient value per knob. Negative gradient = direction to improve.
func TrajectoryGradient(enc *AudioEncoder, patch *Candidate,
	target *TrajectorySpec, targetLFO []float64,
	knobs []KnobDef, note uint8, dur time.Duration,
	cfg TrajectoryConfig) []float64 {

	if enc == nil || target == nil || len(knobs) == 0 {
		return nil
	}

	ctx := context.Background()
	_ = ctx // available for future budget control

	// Render the base patch and extract trajectory
	baseAudio := RenderCandidate(patch, note, dur)
	baseNorm := NormalizeRMS(TrimOrPad(baseAudio, target.TimeSteps*int(dur.Seconds()*float64(RenderSampleRate))))
	baseTraj := ExtractTrajectory(enc, baseNorm)
	if baseTraj == nil {
		return nil
	}

	// Find worst time step t*
	n := target.TimeSteps
	if baseTraj.TimeSteps < n {
		n = baseTraj.TimeSteps
	}
	worstT := 0
	worstD := 0.0
	for t := 0; t < n; t++ {
		dt := cosineDistFloat32(target.Embeddings[t], baseTraj.Embeddings[t])
		if dt > worstD {
			worstD = dt
			worstT = t
		}
	}

	// Compute gradient for each knob via finite differences
	gradient := make([]float64, len(knobs))
	epsilon := 3.0

	for i, k := range knobs {
		shifted := patch.Copy()
		curVal := k.getter(shifted)
		delta := int(epsilon)
		newVal := curVal + delta
		if newVal > k.max {
			newVal = curVal - delta
			if newVal < k.min {
				newVal = curVal
			}
		}
		if newVal == curVal {
			gradient[i] = 0
			continue
		}
		k.setter(shifted, newVal)
		actualDelta := float64(newVal - curVal)

		// Render shifted patch and extract trajectory
		shiftedAudio := RenderCandidate(shifted, note, dur)
		shiftedNorm := NormalizeRMS(TrimOrPad(shiftedAudio, len(baseNorm)))
		shiftedTraj := ExtractTrajectory(enc, shiftedNorm)
		if shiftedTraj == nil {
			gradient[i] = 0
			continue
		}

		// Compute d(t*) at shifted parameter
		if worstT < shiftedTraj.TimeSteps {
			dShifted := cosineDistFloat32(target.Embeddings[worstT], shiftedTraj.Embeddings[worstT])
			gradient[i] = (dShifted - worstD) / actualDelta
		}
	}

	return gradient
}

// distanceTrajectorySpectrum computes the FFT magnitude spectrum of a 1D distance signal.
func distanceTrajectorySpectrum(d []float64, bins int) []float64 {
	n := nextPowerOf2(len(d))
	if bins > n/2 {
		bins = n / 2
	}
	signal := make([]float64, n)
	copy(signal, d)

	spectrum := make([]float64, bins)
	for k := 0; k < bins; k++ {
		re, im := 0.0, 0.0
		for t := 0; t < n; t++ {
			angle := -2.0 * math.Pi * float64(k*t) / float64(n)
			re += signal[t] * math.Cos(angle)
			im += signal[t] * math.Sin(angle)
		}
		spectrum[k] = math.Sqrt(re*re + im*im)
	}
	return spectrum
}

// cosineDistFloat32 computes 1 - cosine similarity between two float32 vectors.
func cosineDistFloat32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range a {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom < 1e-10 {
		// Both vectors are near-zero: distance is 0 if both are zero, 1 otherwise
		if normA < 1e-10 && normB < 1e-10 {
			return 0.0
		}
		return 1.0
	}
	cosSim := dot / denom
	if cosSim > 1.0 {
		cosSim = 1.0
	}
	if cosSim < -1.0 {
		cosSim = -1.0
	}
	return 1.0 - cosSim
}

// cosineDistFloat64 computes 1 - cosine similarity between two float64 vectors.
func cosineDistFloat64(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom < 1e-10 {
		if normA < 1e-10 && normB < 1e-10 {
			return 0.0
		}
		return 1.0
	}
	cosSim := dot / denom
	if cosSim > 1.0 {
		cosSim = 1.0
	}
	if cosSim < -1.0 {
		cosSim = -1.0
	}
	return 1.0 - cosSim
}

// nextPowerOf2 returns the smallest power of 2 >= n.
func nextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

// blendAlpha returns the trajectory weight for the current generation.
func blendAlpha(gen, totalGens int, startAlpha, endAlpha float64) float64 {
	if totalGens <= 0 {
		return startAlpha
	}
	t := float64(gen) / float64(totalGens)
	return startAlpha + t*(endAlpha-startAlpha)
}

// selectTopK returns the indices of the K smallest values in vals.
func selectTopK(vals []float64, k int) []int {
	type idxVal struct {
		idx int
		val float64
	}
	sorted := make([]idxVal, len(vals))
	for i, v := range vals {
		sorted[i] = idxVal{i, v}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].val < sorted[j].val
	})
	if k > len(sorted) {
		k = len(sorted)
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = sorted[i].idx
	}
	return result
}

// argminFloat64 returns the index of the minimum value.
func argminFloat64(vals []float64) int {
	if len(vals) == 0 {
		return 0
	}
	best := 0
	for i, v := range vals {
		if v < vals[best] {
			best = i
		}
	}
	return best
}

// TODO: decouple TrajectoryGradient from Config — pass encoder/trajectory as parameters
func init() {
	// Suppress unused import warning for sort (used by selectTopK)
	_ = fmt.Sprintf
}
