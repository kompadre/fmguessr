package match

import (
	"math"
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

const (
	melSpecSize     = 128
	melSpecNFrames  = 128
	melSpecNHop     = 512
	melSpecNFFT     = 2048
	melSpecFMin     = 20.0
	melSpecFMax     = 8000.0
)

// ComputeMelSpectrogram extracts a 128×128 mel spectrogram from PCM audio.
// The output is normalized to [0, 1] and matches the Python training pipeline exactly.
func ComputeMelSpectrogram(samples []float64, sampleRate int) [melSpecSize][melSpecSize]float64 {
	var result [melSpecSize][melSpecSize]float64

	nSamples := melSpecNFrames*melSpecNHop + melSpecNFFT
	if len(samples) < melSpecNFFT {
		return result
	}
	if len(samples) > nSamples {
		samples = samples[:nSamples]
	}

	// Build mel filterbank
	filterbank := buildMelFilterbank(melSpecSize, melSpecNFFT, sampleRate)

	// Compute STFT frames
	nFrames := (len(samples) - melSpecNFFT) / melSpecNHop + 1
	if nFrames < 1 {
		nFrames = 1
	}

	win := window.Hann(melSpecNFFT)
	spec := make([]float64, melSpecSize*nFrames)

	for i := 0; i < nFrames; i++ {
		start := i * melSpecNHop
		end := start + melSpecNFFT
		if end > len(samples) {
			end = len(samples)
		}

		// Apply window and FFT
		frame := make([]float64, melSpecNFFT)
		n := copy(frame, samples[start:end])
		for j := 0; j < n; j++ {
			frame[j] *= win[j]
		}

		spectrum := fft.FFTReal(frame)
		halfWindow := melSpecNFFT/2 + 1

		// Compute mel energies
		for m := 0; m < melSpecSize; m++ {
			energy := 0.0
			for j := 0; j < halfWindow && j < len(spectrum); j++ {
				mag := cmplx.Abs(spectrum[j])
				energy += filterbank[m][j] * mag
			}
			// Log scale
			spec[i*melSpecSize+m] = math.Log(energy + 1e-5)
		}
	}

	// Normalize to [0, 1]
	specMin := math.MaxFloat64
	specMax := -math.MaxFloat64
	for _, v := range spec {
		if v < specMin {
			specMin = v
		}
		if v > specMax {
			specMax = v
		}
	}

	range_val := specMax - specMin
	if range_val < 1e-8 {
		return result
	}

	for i := 0; i < melSpecSize; i++ {
		for j := 0; j < melSpecSize; j++ {
			idx := j*melSpecSize + i
			if idx < len(spec) {
				result[i][j] = (spec[idx] - specMin) / range_val
			}
		}
	}

	return result
}

// buildMelFilterbank constructs triangular mel-scale filters over FFT bins.
func buildMelFilterbank(nMels, nFFT, sampleRate int) [][]float64 {
	numBins := nFFT/2 + 1
	bank := make([][]float64, nMels)

	melMin := hzToMel(melSpecFMin)
	melMax := hzToMel(melSpecFMax)

	for m := 0; m < nMels; m++ {
		bank[m] = make([]float64, numBins)

		fLeft := melToHz(melMin + float64(m)*(melMax-melMin)/float64(nMels))
		fCenter := melToHz(melMin + float64(m+1)*(melMax-melMin)/float64(nMels))
		fRight := melToHz(melMin + float64(m+2)*(melMax-melMin)/float64(nMels))

		binLeft := int(math.Floor(float64(nFFT+1) * fLeft / float64(sampleRate)))
		binCenter := int(math.Floor(float64(nFFT+1) * fCenter / float64(sampleRate)))
		binRight := int(math.Floor(float64(nFFT+1) * fRight / float64(sampleRate)))

		if binLeft < 0 {
			binLeft = 0
		}
		if binCenter < binLeft {
			binCenter = binLeft + 1
		}
		if binRight <= binCenter {
			binRight = binCenter + 1
		}
		if binRight >= numBins {
			binRight = numBins - 1
		}
		if binCenter >= numBins {
			binCenter = numBins - 1
		}
		if binLeft >= numBins {
			binLeft = numBins - 1
		}

		for k := binLeft; k < binCenter; k++ {
			if binCenter > binLeft {
				bank[m][k] = float64(k-binLeft) / float64(binCenter-binLeft)
			}
		}
		for k := binCenter; k < binRight; k++ {
			if binRight > binCenter {
				bank[m][k] = float64(binRight-k) / float64(binRight-binCenter)
			}
		}
	}

	return bank
}
