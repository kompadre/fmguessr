package match

import (
	"fmt"
	"math"
	"time"
)

// FormantDistance computes the distance between two sets of formant peaks.
// Uses greedy nearest-neighbor matching with weighted frequency + amplitude differences.
//
// Weights: frequency 0.6, amplitude 0.4, unmatched penalty 0.5 per missing formant.
func FormantDistance(targetPeaks, candPeaks []FormantPeak) float64 {
	if len(targetPeaks) == 0 && len(candPeaks) == 0 {
		return 0.0
	}
	if len(targetPeaks) == 0 || len(candPeaks) == 0 {
		return 1.0
	}

	const wFreq = 0.6
	const wAmp = 0.4
	const wUnmatched = 0.5

	used := make([]bool, len(candPeaks))
	totalDist := 0.0
	matched := 0

	for _, tp := range targetPeaks {
		bestIdx := -1
		bestDist := math.MaxFloat64

		for j, cp := range candPeaks {
			if used[j] {
				continue
			}
			// Frequency distance (relative)
			freqDist := math.Abs(tp.Frequency-cp.Frequency) / math.Max(tp.Frequency, 1.0)
			// Amplitude distance
			ampDist := math.Abs(tp.Amplitude - cp.Amplitude)
			// Combined
			dist := wFreq*freqDist + wAmp*ampDist

			if dist < bestDist {
				bestDist = dist
				bestIdx = j
			}
		}

		if bestIdx >= 0 && bestDist < 2.0 {
			used[bestIdx] = true
			totalDist += bestDist
			matched++
		} else {
			// Unmatched target formant
			totalDist += wUnmatched
		}
	}

	n := float64(len(targetPeaks))
	if n == 0 {
		return 0.0
	}

	return totalDist / n
}

// SpectralEnvelopeDistance computes cosine distance between two spectral envelopes.
// Envelopes are log-magnitude arrays from ExtractSpectralEnvelope.
func SpectralEnvelopeDistance(targetEnv, candEnv []float64) float64 {
	if len(targetEnv) == 0 || len(candEnv) == 0 {
		return 1.0
	}

	// Use shorter length
	n := len(targetEnv)
	if len(candEnv) < n {
		n = len(candEnv)
	}

	dot, normA, normB := 0.0, 0.0, 0.0
	for i := 0; i < n; i++ {
		dot += targetEnv[i] * candEnv[i]
		normA += targetEnv[i] * targetEnv[i]
		normB += candEnv[i] * candEnv[i]
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

// GenerateFormantCandidates generates DX7 candidates whose FM sidebands
// best match the target's formant structure.
//
// Strategy:
//  1. Extract target formants
//  2. For representative algorithms, generate candidates with operators dialed to match
//  3. Vary modulator ratios to produce different spectral textures
//  4. Return candidates ranked by formant distance
func GenerateFormantCandidates(targetFormants []FormantPeak, noteFreq float64, note uint8) []*Candidate {
	if len(targetFormants) == 0 || noteFreq <= 0 {
		return nil
	}

	// Use representative algorithms with distinct topologies (not all 32)
	// Group 1: Simple FM (1 carrier + chain) — algo 1, 4, 8
	// Group 2: Parallel carriers — algo 13, 17, 21
	// Group 3: Complex feedback — algo 25, 29, 32
	repAlgos := []uint8{0, 3, 7, 12, 16, 20, 24, 28, 31}

	var candidates []*Candidate
	for _, algo := range repAlgos {
		cand := formantMatchAlgorithm(targetFormants, algo, noteFreq)
		if cand != nil {
			candidates = append(candidates, cand)
		}
	}

	// Sort by formant distance
	if len(candidates) > 1 {
		for i := 0; i < len(candidates); i++ {
			bestIdx := i
			bestDist := FormantDistance(targetFormants, PredictSpectralPeaks(candidates[i], noteFreq))
			for j := i + 1; j < len(candidates); j++ {
				dist := FormantDistance(targetFormants, PredictSpectralPeaks(candidates[j], noteFreq))
				if dist < bestDist {
					bestDist = dist
					bestIdx = j
				}
			}
			if bestIdx != i {
				candidates[i], candidates[bestIdx] = candidates[bestIdx], candidates[i]
			}
		}
	}

	if len(candidates) > 4 {
		candidates = candidates[:4]
	}

	return candidates
}

// formantMatchAlgorithm generates a candidate for a specific algorithm
// with operator parameters tuned to produce the target formants.
func formantMatchAlgorithm(targetFormants []FormantPeak, algo uint8, noteFreq float64) *Candidate {
	if algo >= 32 {
		return nil
	}

	cand := SineBase()
	cand.Algorithm = algo

	// Count available carriers for this algorithm
	carriers := make([]int, 0, 3)
	for opIdx := 0; opIdx < 6; opIdx++ {
		if IsCarrier(algo, opIdx) {
			carriers = append(carriers, opIdx)
		}
	}

	if len(carriers) == 0 {
		return nil
	}

	// Assign formants to carriers — strongest formant to first carrier
	assignedFormants := 0
	for ci, carrierIdx := range carriers {
		if assignedFormants >= len(targetFormants) {
			break
		}

		// Pick strongest unassigned formant
		bestFormantIdx := -1
		bestAmp := 0.0
		for fi, fp := range targetFormants {
			alreadyAssigned := false
			for _, otherCarrier := range carriers[:ci] {
				otherFreq := operatorFreq(cand, otherCarrier, noteFreq)
				if otherFreq > 0 && math.Abs(otherFreq-fp.Frequency)/fp.Frequency < 0.3 {
					alreadyAssigned = true
					break
				}
			}
			if !alreadyAssigned && fp.Amplitude > bestAmp {
				bestAmp = fp.Amplitude
				bestFormantIdx = fi
			}
		}

		if bestFormantIdx < 0 {
			continue
		}

		formantFreq := targetFormants[bestFormantIdx].Frequency

		// Set carrier ratio to match formant frequency
		ratio := formantFreq / noteFreq
		if ratio < 0.5 {
			ratio = 0.5
		}
		if ratio > 31.0 {
			ratio = 31.0
		}

		coarse := int(ratio)
		if coarse < 1 {
			coarse = 1
		}
		if coarse > 31 {
			coarse = 31
		}
		fine := int((ratio/float64(coarse) - 1.0) * 100.0)
		if fine < 0 {
			fine = 0
		}
		if fine > 99 {
			fine = 99
		}

		cand.Ops[carrierIdx].FreqCoarse = uint8(coarse)
		cand.Ops[carrierIdx].FreqFine = uint8(fine)
		cand.Ops[carrierIdx].OutputLevel = 99

		// Set modulator to produce useful sidebands
		// Vary modulator ratio based on algorithm topology for diversity
		modIdx := carrierIdx + 1
		if modIdx < 6 && !IsCarrier(algo, modIdx) {
			// Modulator ratio: different per algorithm group for diversity
			modRatio := 1.0 + float64(algo%4)*0.5 // 1.0, 1.5, 2.0, 2.5
			cand.Ops[modIdx].FreqCoarse = uint8(modRatio)
			cand.Ops[modIdx].FreqFine = 0
			cand.Ops[modIdx].OutputLevel = uint8(60 + (ci*10)%30) // vary modulation depth
		}

		assignedFormants++
	}

	// Feedback varies by algorithm for spectral diversity
	cand.Feedback = uint8(algo % 8)
	if cand.Feedback > 7 {
		cand.Feedback = 7
	}

	return cand
}

// FormantRefine optimizes a candidate on formant distance while constraining
// mel distance to stay within a tolerance of the original.
//
// Strategy: for each knob, try ±1-3 steps. Accept if formant distance improves
// AND mel distance stays within (1+tolerance) × originalMelDist.
// Runs maxIter passes (each pass sweeps all knobs).
func FormantRefine(
	cand *Candidate,
	targetFormants []FormantPeak,
	noteFreq float64,
	note uint8,
	dur time.Duration,
	target []float64,
	rawTargetRMS float64,
	targetMel *PrecomputedTargetMel,
	originalMelDist float64,
	tolerance float64, // e.g. 0.10 = 10%
	maxIter int,
	verbose bool,
) (*Candidate, float64) {
	best := cand.Copy()
	bestFormantDist := FormantDistance(targetFormants, PredictSpectralPeaks(best, noteFreq))
	melCeiling := originalMelDist * (1.0 + tolerance)

	knobs := buildKnobs(true)

	for iter := 0; iter < maxIter; iter++ {
		improved := false

		for _, k := range knobs {
			curVal := k.getter(best)

			// Try ±1, ±2, ±3 steps
			bestDelta := 0
			for _, delta := range []int{-3, -2, -1, 1, 2, 3} {
				newVal := curVal + delta
				if newVal < k.min || newVal > k.max {
					continue
				}

				trial := best.Copy()
				k.setter(trial, newVal)

				// Check formant distance
				trialFormantDist := FormantDistance(targetFormants, PredictSpectralPeaks(trial, noteFreq))
				if trialFormantDist >= bestFormantDist {
					continue
				}

				// Check mel constraint
				trialAudio := RenderCandidate(trial, note, dur)
				trialNorm := NormalizeRMS(TrimOrPad(trialAudio, len(target)))
				trialMelDist := MelCosineDistanceCached(targetMel, trialNorm, RenderSampleRate)

				if trialMelDist <= melCeiling {
					bestFormantDist = trialFormantDist
					bestDelta = delta
					// Don't break — keep trying larger deltas
				}
			}

			if bestDelta != 0 {
				k.setter(best, curVal+bestDelta)
				improved = true
			}
		}

		if !improved {
			break
		}

		if verbose {
			// Check current mel distance
			bestAudio := RenderCandidate(best, note, dur)
			bestNorm := NormalizeRMS(TrimOrPad(bestAudio, len(target)))
			currentMel := MelCosineDistanceCached(targetMel, bestNorm, RenderSampleRate)
			fmt.Printf("[formant-refine] iter %d: formant=%.6f mel=%.6f (ceiling=%.6f)\n",
				iter+1, bestFormantDist, currentMel, melCeiling)
		}
	}

	return best, bestFormantDist
}
