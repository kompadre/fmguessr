package match

import (
	"math"
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/fft"
)

// FingerprintDim is the length of the feature vector used for audio indexing.
const FingerprintDim = 32

// Fingerprint represents a 32-element normalized pitch-invariant acoustic vector.
type Fingerprint [FingerprintDim]float64

// ComputeFingerprint extracts a 32-dimensional pitch-invariant perceptual timbre vector from PCM audio.
func ComputeFingerprint(samples []float64, sampleRate int) Fingerprint {
	var fp Fingerprint
	if len(samples) < 1024 {
		return fp
	}

	// Detect fundamental pitch F0
	f0, conf := detectPitch(samples, sampleRate)
	if f0 < 20.0 || conf < 0.2 {
		f0 = 130.81 // Default C3
	}

	spec := computeSpectrum(samples, sampleRate)

	// 1. Compute Pitch-Normalized Harmonic Profile (10 relative harmonic bins)
	// Harmonic multipliers: 1x, 1.5x, 2x, 3x, 4x, 5x, 6x, 7x, 8x, 10x F0
	harmonicRatios := []float64{1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0}
	if spec != nil {
		totEnergy := 0.0
		for _, v := range spec {
			totEnergy += v * v
		}
		if totEnergy > 1e-10 {
			binWidth := float64(sampleRate) / float64((len(spec)-1)*2)
			for idx, ratio := range harmonicRatios {
				targetHz := f0 * ratio
				centerBin := int(math.Round(targetHz / binWidth))
				hEnergy := 0.0
				// Sum 3 bins around target harmonic
				for b := centerBin - 1; b <= centerBin+1; b++ {
					if b >= 0 && b < len(spec) {
						hEnergy += spec[b] * spec[b]
					}
				}
				fp[idx] = math.Sqrt(hEnergy / totEnergy)
			}
		}
	}

	// 2. Compute Pitch-Normalized MFCCs (13 coefficients scaled relative to F0)
	mfccs := computePitchNormalizedMFCCs(samples, sampleRate, f0, 13)
	for i := 0; i < 13 && i < len(mfccs); i++ {
		fp[10+i] = mfccs[i]
	}

	// 3. Compute 4-frame temporal Pitch-Normalized Spectral Centroids (Centroid / F0)
	numFrames := 4
	frameLen := len(samples) / numFrames
	if frameLen >= 512 {
		var prevMag []float64
		for f := 0; f < numFrames; f++ {
			frame := samples[f*frameLen : (f+1)*frameLen]
			mag := computeSpectrum(frame, sampleRate)
			if mag == nil {
				continue
			}

			// Spectral Centroid normalized by F0 (multiples of fundamental pitch)
			cHz := spectralCentroid(mag, sampleRate)
			fp[23+f] = math.Min(cHz/f0/16.0, 1.0)

			// Spectral Flux (spectral difference from previous frame)
			flux := 0.0
			if prevMag != nil && len(prevMag) == len(mag) {
				for i := range mag {
					diff := mag[i] - prevMag[i]
					if diff > 0 {
						flux += diff * diff
					}
				}
				flux = math.Sqrt(flux) / float64(len(mag))
			}
			if f < 2 {
				fp[27+f] = math.Min(flux*10.0, 1.0)
			}
			prevMag = mag
		}
	}

	// 4. Envelope & Harmonicity features
	feat := ExtractFeatures(samples, sampleRate)
	fp[29] = feat.Harmonicity
	fp[30] = math.Min(feat.AttackMs/1000.0, 1.0)
	fp[31] = feat.SustainLevel

	// L2 Normalize fingerprint vector
	norm := 0.0
	for _, v := range fp {
		norm += v * v
	}
	if norm > 1e-9 {
		norm = math.Sqrt(norm)
		for i := range fp {
			fp[i] /= norm
		}
	}

	return fp
}

// CosineDistance computes 1.0 - CosineSimilarity between two fingerprints (0.0 = identical).
func (f1 Fingerprint) CosineDistance(f2 Fingerprint) float64 {
	dot := 0.0
	for i := 0; i < FingerprintDim; i++ {
		dot += f1[i] * f2[i]
	}
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}
	return 1.0 - dot
}

// computePitchNormalizedMFCCs constructs Mel filters scaled relative to F0 (0.5*F0 to 32*F0)
func computePitchNormalizedMFCCs(samples []float64, sampleRate int, f0 float64, numCoeffs int) []float64 {
	frameSize := 4096
	hopSize := 1024
	numFilters := 26

	if len(samples) < frameSize {
		return make([]float64, numCoeffs)
	}

	minHz := f0 * 0.5
	maxHz := f0 * 32.0
	if maxHz > float64(sampleRate)/2.0 {
		maxHz = float64(sampleRate) / 2.0
	}
	if minHz < 20.0 {
		minHz = 20.0
	}

	filterBank := buildRelativeMelFilterBank(numFilters, frameSize, sampleRate, minHz, maxHz)
	sumCoeffs := make([]float64, numCoeffs)
	frameCount := 0

	win := getHannWindow(frameSize)

	for start := 0; start+frameSize <= len(samples); start += hopSize {
		frame := samples[start : start+frameSize]

		fftIn := make([]complex128, frameSize)
		for i := 0; i < frameSize; i++ {
			fftIn[i] = complex(frame[i]*win[i], 0)
		}
		spectrum := fft.FFT(fftIn)

		power := make([]float64, frameSize/2+1)
		for i := range power {
			mag := cmplx.Abs(spectrum[i])
			power[i] = (mag * mag) / float64(frameSize)
		}

		filterEnergies := make([]float64, numFilters)
		for f := 0; f < numFilters; f++ {
			e := 0.0
			for k := 0; k < len(power); k++ {
				e += power[k] * filterBank[f][k]
			}
			if e < 1e-10 {
				e = 1e-10
			}
			filterEnergies[f] = math.Log(e)
		}

		for i := 0; i < numCoeffs; i++ {
			c := 0.0
			for f := 0; f < numFilters; f++ {
				c += filterEnergies[f] * math.Cos(float64(i)*(float64(f)+0.5)*math.Pi/float64(numFilters))
			}
			sumCoeffs[i] += c
		}
		frameCount++
	}

	if frameCount > 0 {
		for i := range sumCoeffs {
			sumCoeffs[i] /= float64(frameCount)
		}
	}

	return sumCoeffs
}

func buildRelativeMelFilterBank(numFilters, fftSize, sampleRate int, minHz, maxHz float64) [][]float64 {
	numBins := fftSize/2 + 1
	bank := make([][]float64, numFilters)

	minMel := hzToMel(minHz)
	maxMel := hzToMel(maxHz)
	melPoints := make([]float64, numFilters+2)

	for i := range melPoints {
		melPoints[i] = minMel + float64(i)*(maxMel-minMel)/float64(numFilters+1)
	}

	binPoints := make([]int, len(melPoints))
	for i, m := range melPoints {
		hz := melToHz(m)
		binPoints[i] = int(math.Floor(float64(fftSize+1) * hz / float64(sampleRate)))
		if binPoints[i] >= numBins {
			binPoints[i] = numBins - 1
		}
	}

	for f := 0; f < numFilters; f++ {
		bank[f] = make([]float64, numBins)
		start := binPoints[f]
		center := binPoints[f+1]
		end := binPoints[f+2]

		for k := start; k < center; k++ {
			if center > start {
				bank[f][k] = float64(k-start) / float64(center-start)
			}
		}
		for k := center; k < end; k++ {
			if end > center {
				bank[f][k] = float64(end-k) / float64(end-center)
			}
		}
	}

	return bank
}

func hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}
