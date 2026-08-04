package match

import (
	"math"
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

const (
	stftWindowSize = 2048
	stftHopSize    = 512
)

// SpectralDistance computes a combined linear & logarithmic spectral distance at a 1024 scale.
// Lower values indicate more similarity. Both signals should be the same length
// and sampled at the same rate.
func SpectralDistance(target, candidate []float64) float64 {
	return spectralDistanceAtScale(target, candidate, 1024, 256)
}

// PrecomputedTargetSTFT holds a cached target spectrogram for use with SpectralDistanceCached.
type PrecomputedTargetSTFT struct {
	Mag      [][]float64 // STFT magnitude frames
	FrameRMS []float64   // Per-frame RMS energy
	MaxRMS   float64     // Maximum frame RMS
}

// ComputeTargetSTFT precomputes the target's STFT magnitude spectrogram and frame energies.
// Call this once per target and reuse across all candidate evaluations.
func ComputeTargetSTFT(target []float64) *PrecomputedTargetSTFT {
	mag := stftMagnitude(target, 1024, 256)
	rows := len(mag)
	if rows == 0 {
		return &PrecomputedTargetSTFT{Mag: mag}
	}
	cols := len(mag[0])

	frameRMS := make([]float64, rows)
	maxRMS := 0.0
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			v := mag[i][j]
			sum += v * v
		}
		rms := math.Sqrt(sum / float64(cols))
		frameRMS[i] = rms
		if rms > maxRMS {
			maxRMS = rms
		}
	}

	return &PrecomputedTargetSTFT{
		Mag:      mag,
		FrameRMS: frameRMS,
		MaxRMS:   maxRMS,
	}
}

// SpectralDistanceCached computes spectral distance using a precomputed target STFT.
// This avoids recomputing the target spectrogram on every call, giving ~50% speedup
// when evaluating many candidates against the same target.
func SpectralDistanceCached(cached *PrecomputedTargetSTFT, candidate []float64) float64 {
	candidateMag := stftMagnitude(candidate, 1024, 256)

	rows := len(cached.Mag)
	if len(candidateMag) < rows {
		rows = len(candidateMag)
	}
	if rows == 0 {
		return 0
	}
	cols := len(cached.Mag[0])
	if len(candidateMag[0]) < cols {
		cols = len(candidateMag[0])
	}
	if cols == 0 {
		return 0
	}

	var diffNorm, targetNorm float64
	var logDiffNorm, logTargetNorm float64

	const epsilon = 1e-5

	for i := 0; i < rows; i++ {
		weight := 1.0
		if cached.MaxRMS > 1e-10 {
			weight = cached.FrameRMS[i] / cached.MaxRMS
			if weight < 0.1 {
				weight = 0.1
			}
		}

		for j := 0; j < cols; j++ {
			tVal := cached.Mag[i][j]
			cVal := candidateMag[i][j]

			d := tVal - cVal
			diffNorm += d * d * weight
			targetNorm += tVal * tVal * weight

			logT := math.Log(tVal + epsilon)
			logC := math.Log(cVal + epsilon)
			ld := logT - logC
			logDiffNorm += ld * ld * weight
			logTargetNorm += logT * logT * weight
		}
	}

	linearDist := 0.0
	if targetNorm > 1e-10 {
		linearDist = math.Sqrt(diffNorm) / math.Sqrt(targetNorm)
	}

	logDist := 0.0
	if logTargetNorm > 1e-10 {
		logDist = math.Sqrt(logDiffNorm) / math.Sqrt(logTargetNorm)
	}

	return linearDist + 1.0*logDist
}

// SustainSpectralDistance compares only the sustain portions of target and candidate,
// eliminating differences from attack transients and release tails.
// Uses ExtractTimbreWindow to isolate the stable sustain region of each signal,
// then computes Euclidean distance on the 256-bin spectra.
// Returns 0.0 for identical sustain spectra, higher for different ones.
func SustainSpectralDistance(target, candidate []float64, sampleRate int) float64 {
	targetSpec := ExtractTimbreWindow(target, sampleRate)
	candidateSpec := ExtractTimbreWindow(candidate, sampleRate)

	// Euclidean distance on 256-bin spectra (already normalized to [0,1])
	sumSq := 0.0
	for i := 0; i < TimbreSpecBins; i++ {
		d := targetSpec[i] - candidateSpec[i]
		sumSq += d * d
	}
	return math.Sqrt(sumSq)
}

// CosineTimbreDistance compares timbre spectra using cosine distance,
// which is robust to overall magnitude differences between rendering paths.
// Returns 0.0 for identical spectral shapes, 2.0 for opposite.
func CosineTimbreDistance(target, candidate []float64, sampleRate int) float64 {
	targetSpec := ExtractTimbreWindow(target, sampleRate)
	candidateSpec := ExtractTimbreWindow(candidate, sampleRate)

	var dotProduct, normA, normB float64
	for i := 0; i < TimbreSpecBins; i++ {
		dotProduct += targetSpec[i] * candidateSpec[i]
		normA += targetSpec[i] * targetSpec[i]
		normB += candidateSpec[i] * candidateSpec[i]
	}

	if normA < 1e-10 || normB < 1e-10 {
		return 2.0 // maximally different
	}

	cosSim := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	// Clamp to [-1, 1] to avoid floating point issues
	if cosSim > 1.0 {
		cosSim = 1.0
	}
	if cosSim < -1.0 {
		cosSim = -1.0
	}

	// Convert cosine similarity to distance: 0 = identical, 2 = opposite
	return 1.0 - cosSim
}

// ParameterPenalty computes a multiplicative penalty when DX7 patch parameters
// diverge catastrophically from the target. Returns 1.0 when all params are close,
// higher values when params diverge beyond a threshold.
// penalty = 1.0 + sum(max(0, (|Δnorm| - threshold) * scale)²)
func ParameterPenalty(targetParams, candidateParams [DX7LabelDims]float64) float64 {
	const threshold = 0.8 // Normalized diff threshold before penalty kicks in
	const scale = 2.0    // Penalty scaling factor

	penalty := 1.0
	for i := 0; i < DX7LabelDims; i++ {
		diff := math.Abs(targetParams[i] - candidateParams[i])
		if diff > threshold {
			excess := (diff - threshold) * scale
			penalty += excess * excess
		}
	}
	return penalty
}

// CombinedDistance computes a distance that combines cosine spectral similarity
// with a parameter penalty. The penalty multiplies the distance when any parameter
// is catastrophically wrong, separating true matches from near-misses.
// Returns 0.0 for perfect matches, higher for worse matches.
func CombinedDistance(target, candidate []float64, sampleRate int,
	targetParams, candidateParams [DX7LabelDims]float64) float64 {

	spectralDist := CosineTimbreDistance(target, candidate, sampleRate)
	paramPenalty := ParameterPenalty(targetParams, candidateParams)

	return spectralDist * paramPenalty
}

// TemporalSpectralDistance computes a multi-stage temporal distance comparing
// target and candidate PCM signals using TemporalTimbreSpec.
func TemporalSpectralDistance(target, candidate []float64, sampleRate int) float64 {
	targetSpec := ExtractTemporalTimbreSpec(target, sampleRate)
	candidateSpec := ExtractTemporalTimbreSpec(candidate, sampleRate)
	return TemporalTimbreDistance(targetSpec, candidateSpec)
}

// CalcDBPenalty calculates a fitness penalty if the un-normalized candidate RMS
// is significantly quieter than the un-normalized target RMS (> 6.0 dB quieter).
func CalcDBPenalty(rawCandRMS, targetRMS float64) float64 {
	if rawCandRMS < 1e-4 {
		return 10.0 // Hard discard / heavy penalty for near-silent candidates
	}
	if targetRMS < 1e-4 {
		return 0.0 // Target itself is silent
	}

	candDB := 20.0 * math.Log10(rawCandRMS)
	targetDB := 20.0 * math.Log10(targetRMS)
	dropDB := targetDB - candDB

	// Allow candidate to be up to 6.0 dB quieter without penalty.
	// Beyond 6.0 dB drop, apply penalty proportional to dB drop.
	if dropDB > 6.0 {
		return (dropDB - 6.0) / 6.0 // e.g. 12 dB drop = +1.0 penalty, 18 dB drop = +2.0 penalty
	}
	return 0.0
}

func spectralDistanceAtScale(target, candidate []float64, winSize, hopSize int) float64 {
	targetMag := stftMagnitude(target, winSize, hopSize)
	candidateMag := stftMagnitude(candidate, winSize, hopSize)

	// Ensure same dimensions
	rows := len(targetMag)
	if len(candidateMag) < rows {
		rows = len(candidateMag)
	}
	if rows == 0 {
		return 0
	}
	cols := len(targetMag[0])
	if len(candidateMag[0]) < cols {
		cols = len(candidateMag[0])
	}
	if cols == 0 {
		return 0
	}

	// Frobenius norm of difference and target with frame-energy weighting.
	// Weighting active frames higher than silent tail frames prevents trailing silence
	// from swamping active sound matching or favoring silent candidates.
	var diffNorm, targetNorm float64
	var logDiffNorm, logTargetNorm float64

	const epsilon = 1e-5

	// Compute per-frame target RMS energy and max frame energy
	frameRMS := make([]float64, rows)
	maxRMS := 0.0
	for i := 0; i < rows; i++ {
		sum := 0.0
		for j := 0; j < cols; j++ {
			v := targetMag[i][j]
			sum += v * v
		}
		rms := math.Sqrt(sum / float64(cols))
		frameRMS[i] = rms
		if rms > maxRMS {
			maxRMS = rms
		}
	}

	for i := 0; i < rows; i++ {
		weight := 1.0
		if maxRMS > 1e-10 {
			weight = frameRMS[i] / maxRMS
			if weight < 0.1 {
				weight = 0.1
			}
		}

		for j := 0; j < cols; j++ {
			tVal := targetMag[i][j]
			cVal := candidateMag[i][j]

			// Linear Distance component
			d := tVal - cVal
			diffNorm += d * d * weight
			targetNorm += tVal * tVal * weight

			// Logarithmic Distance component
			logT := math.Log(tVal + epsilon)
			logC := math.Log(cVal + epsilon)
			ld := logT - logC
			logDiffNorm += ld * ld * weight
			logTargetNorm += logT * logT * weight
		}
	}

	linearDist := 0.0
	if targetNorm > 1e-10 {
		linearDist = math.Sqrt(diffNorm) / math.Sqrt(targetNorm)
	}

	logDist := 0.0
	if logTargetNorm > 1e-10 {
		logDist = math.Sqrt(logDiffNorm) / math.Sqrt(logTargetNorm)
	}

	// Linear + Log-Spectral distance
	return linearDist + 1.0*logDist
}

// stftMagnitude computes the STFT magnitude spectrogram.
func stftMagnitude(samples []float64, winSize, hopSize int) [][]float64 {
	nFrames := (len(samples)-winSize)/hopSize + 1
	if nFrames < 1 {
		nFrames = 1
	}
	halfWindow := winSize/2 + 1

	win := window.Hann(winSize)
	mags := make([][]float64, nFrames)

	for i := 0; i < nFrames; i++ {
		start := i * hopSize
		end := start + winSize
		if end > len(samples) {
			end = len(samples)
		}

		frame := make([]float64, winSize)
		n := copy(frame, samples[start:end])
		// Apply window
		for j := 0; j < n; j++ {
			frame[j] *= win[j]
		}

		// FFT
		spectrum := fft.FFTReal(frame)

		// Take magnitude of first half (positive frequencies)
		mag := make([]float64, halfWindow)
		for j := 0; j < halfWindow && j < len(spectrum); j++ {
			mag[j] = cmplx.Abs(spectrum[j])
		}
		mags[i] = mag
	}

	return mags
}
