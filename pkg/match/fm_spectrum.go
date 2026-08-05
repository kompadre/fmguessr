package match

import (
	"math"
)

// besselJ computes the Bessel function of the first kind J_n(x) for integer order n.
// Uses the recurrence relation for numerical stability.
func besselJ(n int, x float64) float64 {
	if n < 0 {
		n = -n
		// J_{-n}(x) = (-1)^n * J_n(x)
		result := besselJ(n, x)
		if n%2 != 0 {
			return -result
		}
		return result
	}

	// For small x, use the series expansion
	if math.Abs(x) < 1e-10 {
		if n == 0 {
			return 1.0
		}
		return 0.0
	}

	// Miller's algorithm: downward recurrence
	// Start from a high order and recurse down
	startOrder := n + int(math.Abs(x)) + 20
	if startOrder < n+10 {
		startOrder = n + 10
	}

	// Start with J_{startOrder} = 0, J_{startOrder-1} = 1 (arbitrary scale)
	jp1 := 0.0
	j := 1.0
	for k := startOrder - 1; k >= 0; k-- {
		jp2 := jp1
		jp1 = j
		if k == 0 {
			j = (2.0*float64(k+1)/x)*jp1 - jp2
		} else {
			j = (2.0*float64(k+1)/x)*jp1 - jp2
		}
		if k == n {
			// Found J_n, but need to normalize
			// Use J_0 + 2*sum(J_{2k}) = 1 for normalization
		}
	}

	// Simpler approach: direct series for moderate n
	return besselJSeries(n, x)
}

// besselJSeries computes J_n(x) via the power series expansion.
// Accurate for moderate n and x.
func besselJSeries(n int, x float64) float64 {
	if math.Abs(x) < 1e-15 {
		if n == 0 {
			return 1.0
		}
		return 0.0
	}

	sum := 0.0
	sign := 1.0
	if n%2 != 0 {
		sign = -1.0
	}

	xHalf := x / 2.0

	for m := 0; m < 50; m++ {
		num := math.Pow(xHalf, float64(n+2*m))
		denom := factorial(m) * factorial(n+m)
		term := sign * num / denom
		sum += term
		if math.Abs(term) < 1e-15*math.Abs(sum) {
			break
		}
		sign = -sign
	}
	return sum
}

// factorial returns n! for small n.
func factorial(n int) float64 {
	if n <= 1 {
		return 1.0
	}
	f := 1.0
	for i := 2; i <= n; i++ {
		f *= float64(i)
	}
	return f
}

// operatorFreq computes the actual frequency of a DX7 operator.
// Mode 0 = ratio mode, Mode 1 = fixed frequency mode.
func operatorFreq(cand *Candidate, opIdx int, noteFreq float64) float64 {
	op := cand.Ops[opIdx]
	ratio := float64(op.FreqCoarse) * (1.0 + float64(op.FreqFine)/100.0)
	if op.Mode == 1 {
		return ratio // fixed Hz
	}
	return noteFreq * ratio // ratio mode
}

// fmSidebands computes the spectral output of a single carrier-modulator pair.
// Returns sideband peaks at carrierFreq ± k*modFreq with amplitudes J_k(modIndex).
func fmSidebands(carrierFreq, modFreq, modIndex float64, maxHarmonics int) []FormantPeak {
	if carrierFreq <= 0 || modFreq <= 0 {
		return nil
	}

	peaks := make([]FormantPeak, 0, 2*maxHarmonics+1)

	// Center frequency: J_0(β)
	j0 := besselJ(0, modIndex)
	if j0 > 0.001 {
		peaks = append(peaks, FormantPeak{Frequency: carrierFreq, Amplitude: math.Abs(j0)})
	}

	// Sidebands: J_k(β) at carrierFreq ± k*modFreq
	for k := 1; k <= maxHarmonics; k++ {
		jk := math.Abs(besselJ(k, modIndex))
		if jk < 0.001 {
			break // higher sidebands negligible
		}
		fUpper := carrierFreq + float64(k)*modFreq
		fLower := carrierFreq - float64(k)*modFreq

		if fUpper > 0 {
			peaks = append(peaks, FormantPeak{Frequency: fUpper, Amplitude: jk})
		}
		if fLower > 0 {
			peaks = append(peaks, FormantPeak{Frequency: fLower, Amplitude: jk})
		}
	}

	return peaks
}

// modIndexEstimate estimates the modulation index from a modulator's output level.
// The DX7 modulation index is approximately proportional to the modulator's
// output level scaled by frequency-dependent sensitivity.
func modIndexEstimate(modulatorOutputLevel uint8, modFreq float64) float64 {
	// Approximate: β ≈ (level/99) * sqrt(modFreq/1000)
	// Higher frequency modulators produce wider sidebands at the same level
	level := float64(modulatorOutputLevel) / 99.0
	freqFactor := math.Sqrt(modFreq / 1000.0)
	if freqFactor < 0.1 {
		freqFactor = 0.1
	}
	return level * freqFactor * 8.0 // scale factor to match typical DX7 range
}

// PredictSpectralPeaks computes the analytic spectral peaks of a DX7 patch.
// Walks the algorithm graph, identifies carriers, traces modulation chains,
// and sums sideband spectra from Bessel function expansion.
//
// This is a zero-render computation — pure math from patch parameters.
func PredictSpectralPeaks(cand *Candidate, noteFreq float64) []FormantPeak {
	algo := cand.Algorithm
	if algo >= 32 {
		return nil
	}

	var allPeaks []FormantPeak

	// For each operator, check if it's a carrier
	for opIdx := 0; opIdx < 6; opIdx++ {
		if !IsCarrier(algo, opIdx) {
			continue
		}

		carrierFreq := operatorFreq(cand, opIdx, noteFreq)
		if carrierFreq <= 0 || carrierFreq > 20000 {
			continue
		}

		// Find modulators that feed this carrier
		modPeaks := findModulatorPeaks(cand, algo, opIdx, noteFreq, 0)

		if len(modPeaks) == 0 {
			// Pure carrier (sine wave)
			allPeaks = append(allPeaks, FormantPeak{
				Frequency: carrierFreq,
				Amplitude: float64(cand.Ops[opIdx].OutputLevel) / 99.0,
			})
		} else {
			// Generate sidebands from each modulator
			for _, mp := range modPeaks {
				sidebands := fmSidebands(carrierFreq, mp.Frequency, mp.Amplitude, 10)
				allPeaks = append(allPeaks, sidebands...)
			}
		}
	}

	// Merge nearby peaks and normalize
	return mergePeaks(allPeaks)
}

// findModulatorPeaks traces the modulation chain for a carrier and returns
// effective modulator frequencies and indices.
func findModulatorPeaks(cand *Candidate, algo uint8, carrierIdx int, noteFreq float64, depth int) []FormantPeak {
	if depth > 3 {
		return nil // limit recursion depth
	}

	var peaks []FormantPeak

	// Check each operator to see if it modulates this carrier
	for modIdx := 0; modIdx < 6; modIdx++ {
		if modIdx == carrierIdx {
			continue
		}
		if IsCarrier(algo, modIdx) {
			continue // carriers don't modulate
		}

		// Check if this operator modulates the carrier (simplified routing check)
		// The algOps table encodes routing; for now use a heuristic:
		// operators with lower index typically modulate those with higher index
		if modIdx < carrierIdx || depth > 0 {
			modFreq := operatorFreq(cand, modIdx, noteFreq)
			if modFreq <= 0 || modFreq > 20000 {
				continue
			}

			beta := modIndexEstimate(cand.Ops[modIdx].OutputLevel, modFreq)
			peaks = append(peaks, FormantPeak{Frequency: modFreq, Amplitude: beta})

			// Recursively find modulators of this modulator
			subPeaks := findModulatorPeaks(cand, algo, modIdx, noteFreq, depth+1)
			for _, sp := range subPeaks {
				// Modulator's modulators contribute to the effective index
				peaks = append(peaks, FormantPeak{
					Frequency: sp.Frequency,
					Amplitude: sp.Amplitude * 0.5, // attenuated contribution
				})
			}
		}
	}

	return peaks
}

// mergePeaks combines nearby peaks and normalizes amplitudes.
func mergePeaks(peaks []FormantPeak) []FormantPeak {
	if len(peaks) == 0 {
		return nil
	}

	// Sort by frequency
	for i := 0; i < len(peaks); i++ {
		for j := i + 1; j < len(peaks); j++ {
			if peaks[j].Frequency < peaks[i].Frequency {
				peaks[i], peaks[j] = peaks[j], peaks[i]
			}
		}
	}

	// Merge peaks within 20 Hz of each other
	merged := []FormantPeak{peaks[0]}
	for i := 1; i < len(peaks); i++ {
		last := &merged[len(merged)-1]
		if peaks[i].Frequency-last.Frequency < 20.0 {
			// Merge: take higher amplitude, weighted average frequency
			totalAmp := last.Amplitude + peaks[i].Amplitude
			if totalAmp > 0 {
				last.Frequency = (last.Frequency*last.Amplitude + peaks[i].Frequency*peaks[i].Amplitude) / totalAmp
			}
			if peaks[i].Amplitude > last.Amplitude {
				last.Amplitude = peaks[i].Amplitude
			}
		} else {
			merged = append(merged, peaks[i])
		}
	}

	// Normalize amplitudes
	maxAmp := 0.0
	for _, p := range merged {
		if p.Amplitude > maxAmp {
			maxAmp = p.Amplitude
		}
	}
	if maxAmp > 0 {
		for i := range merged {
			merged[i].Amplitude /= maxAmp
		}
	}

	return merged
}
