package match

import (
	"math"
	"math/cmplx"
	"sort"

	"github.com/madelynnblue/go-dsp/fft"
	"github.com/madelynnblue/go-dsp/window"
)

const (
	TimbreSpecBins = 256   // compressed timbral spectrum size
	TimbreWinMs    = 300   // stable window duration in ms
	TimbreSkipMs   = 80    // skip attack transient in ms
	TimbreNFrames  = 128   // STFT frames for full spectrogram
)

// ExtractTimbreWindow extracts a clean sustain-only timbral fingerprint from PCM audio.
// Algorithm:
//  1. Skip attack transient (first TimbreSkipMs)
//  2. Find the TimbreWinMs window with minimum spectral flux (most stable sustain)
//  3. Normalize frame energies to remove volume decay
//  4. Compute temporal median STFT to collapse vibrato & detune beating
//  5. Return compressed 256-bin timbral spectrum normalized to [0,1]
func ExtractTimbreWindow(samples []float64, sampleRate int) [TimbreSpecBins]float64 {
	var result [TimbreSpecBins]float64

	if len(samples) < 1024 {
		return result
	}

	// Find active signal range
	maxPeak := 0.0
	for _, s := range samples {
		if abs := math.Abs(s); abs > maxPeak {
			maxPeak = abs
		}
	}

	skipSamples := int(float64(sampleRate) * float64(TimbreSkipMs) / 1000.0)
	winSamples := int(float64(sampleRate) * float64(TimbreWinMs) / 1000.0)

	if maxPeak > 1e-4 {
		thresh := maxPeak * 0.01
		activeStart := 0
		for i, s := range samples {
			if math.Abs(s) >= thresh {
				activeStart = i
				break
			}
		}
		activeEnd := len(samples)
		for i := len(samples) - 1; i >= activeStart; i-- {
			if math.Abs(samples[i]) >= thresh {
				activeEnd = i + 1
				break
			}
		}
		activeLen := activeEnd - activeStart

		// If total active sound is short (< 350ms, e.g. hi-hats / percussive hits),
		// start at the active attack transient and use the active duration window.
		if activeLen < int(float64(sampleRate)*0.35) {
			skipSamples = activeStart
			winSamples = activeLen
			if winSamples > int(float64(sampleRate)*0.30) {
				winSamples = int(float64(sampleRate)*0.30)
			}
			if winSamples < 1024 {
				winSamples = 1024
			}
		}
	}

	if skipSamples >= len(samples) {
		skipSamples = 0
	}
	if skipSamples+winSamples > len(samples) {
		winSamples = len(samples) - skipSamples
	}
	if winSamples < 1024 {
		return result
	}

	sustain := samples[skipSamples : skipSamples+winSamples]

	// 1. Compute spectral flux per frame to find the most stable window
	fluxWin := 1024
	fluxHop := 512
	nFluxFrames := (len(sustain) - fluxWin) / fluxHop
	if nFluxFrames < 2 {
		nFluxFrames = 1
	}

	fluxes := make([]float64, nFluxFrames)
	var prevMag []float64
	for i := 0; i < nFluxFrames; i++ {
		start := i * fluxHop
		frame := sustain[start : start+fluxWin]
		mag := fftMag(frame)
		if prevMag != nil {
			flux := 0.0
			for j := range mag {
				d := mag[j] - prevMag[j]
				if d > 0 {
					flux += d * d
				}
			}
			fluxes[i] = math.Sqrt(flux)
		}
		prevMag = mag
	}

	// 2. Find window of TimbreWinMs with minimum total flux
	targetWinSamples := winSamples
	bestStart := 0
	bestFlux := math.MaxFloat64

	frameSize := fluxWin
	hopSize := fluxHop
	for i := 0; i+frameSize <= len(sustain); i += hopSize {
		totalFlux := 0.0
		framesInWindow := 0
		for j := i; j+frameSize <= len(sustain) && j < i+targetWinSamples; j += hopSize {
			idx := j / hopSize
			if idx < len(fluxes) {
				totalFlux += fluxes[idx]
				framesInWindow++
			}
		}
		if framesInWindow > 0 {
			totalFlux /= float64(framesInWindow)
		}
		if totalFlux < bestFlux {
			bestFlux = totalFlux
			bestStart = i
		}
	}

	// Extract the stable window
	end := bestStart + targetWinSamples
	if end > len(sustain) {
		end = len(sustain)
	}
	stable := sustain[bestStart:end]

	// 3. Normalize frame energies (de-trend decaying volume)
	frameLen := 512
	nFrames := len(stable) / frameLen
	if nFrames < 1 {
		nFrames = 1
	}

	normFrames := make([][]float64, nFrames)
	for i := 0; i < nFrames; i++ {
		start := i * frameLen
		end := start + frameLen
		if end > len(stable) {
			end = len(stable)
		}
		frame := stable[start:end]

		// Compute frame RMS
		rms := 0.0
		for _, v := range frame {
			rms += v * v
		}
		rms = math.Sqrt(rms / float64(len(frame)))
		if rms < 1e-10 {
			rms = 1e-10
		}

		// Normalize frame to unit RMS
		normFrame := make([]float64, len(frame))
		for j, v := range frame {
			normFrame[j] = v / rms
		}
		normFrames[i] = normFrame
	}

	// 4. Compute temporal median STFT (collapse vibrato & detune beating)
	nFFT := 1024
	halfWin := nFFT/2 + 1
	win := window.Hann(nFFT)

	// Collect STFT magnitudes for each frequency bin across frames
	allMags := make([][]float64, nFrames)
	for i := 0; i < nFrames; i++ {
		frame := normFrames[i]
		// Pad if needed
		if len(frame) < nFFT {
			padded := make([]float64, nFFT)
			copy(padded, frame)
			frame = padded
		}

		fftIn := make([]complex128, nFFT)
		for j := 0; j < nFFT; j++ {
			fftIn[j] = complex(frame[j]*win[j], 0)
		}
		spectrum := fft.FFT(fftIn)

		mag := make([]float64, halfWin)
		for j := 0; j < halfWin; j++ {
			mag[j] = cmplx.Abs(spectrum[j])
		}
		allMags[i] = mag
	}

	// Compute median across frames for each frequency bin
	medianSpec := make([]float64, halfWin)
	for j := 0; j < halfWin; j++ {
		vals := make([]float64, nFrames)
		for i := 0; i < nFrames; i++ {
			if j < len(allMags[i]) {
				vals[i] = allMags[i][j]
			}
		}
		sort.Float64s(vals)
		medianSpec[j] = vals[nFrames/2]
	}

	// 5. Compress to TimbreSpecBins (256) using mean+variance per bin
	// Mean captures average spectrum; variance captures temporal modulation
	// that distinguishes different FM algorithm topologies
	// Limit to ~8kHz — DX7 hardware filters above this
	maxBin := int(float64(halfWin) * 8000.0 / float64(RenderSampleRate))
	if maxBin < TimbreSpecBins/2 {
		maxBin = TimbreSpecBins / 2
	}
	if maxBin > halfWin {
		maxBin = halfWin
	}

	halfBins := TimbreSpecBins / 2
	binsPerOutput := maxBin / halfBins
	if binsPerOutput < 1 {
		binsPerOutput = 1
	}

	// First half: mean spectrum
	for i := 0; i < halfBins; i++ {
		sum := 0.0
		count := 0
		for j := i * binsPerOutput; j < (i+1)*binsPerOutput && j < maxBin; j++ {
			sum += medianSpec[j]
			count++
		}
		if count > 0 {
			result[i] = sum / float64(count)
		}
	}

	// Second half: variance spectrum (captures temporal modulation)
	for i := 0; i < halfBins; i++ {
		sumSq := 0.0
		sum := 0.0
		count := 0
		for j := i * binsPerOutput; j < (i+1)*binsPerOutput && j < maxBin; j++ {
			for f := 0; f < nFrames; f++ {
				if j < len(allMags[f]) {
					v := allMags[f][j]
					sum += v
					sumSq += v * v
				}
			}
			count += nFrames
		}
		if count > 1 {
			mean := sum / float64(count)
			variance := sumSq/float64(count) - mean*mean
			if variance < 0 {
				variance = 0
			}
			result[halfBins+i] = math.Sqrt(variance)
		}
	}

	// Normalize to [0, 1]
	specMin := math.MaxFloat64
	specMax := -math.MaxFloat64
	for _, v := range result {
		if v < specMin {
			specMin = v
		}
		if v > specMax {
			specMax = v
		}
	}
	rangeVal := specMax - specMin
	if rangeVal > 1e-8 {
		for i := range result {
			result[i] = (result[i] - specMin) / rangeVal
		}
	}

	return result
}

// fftMag computes the magnitude spectrum of a signal frame.
func fftMag(samples []float64) []float64 {
	n := len(samples)
	if n < 64 {
		return nil
	}

	// Pad to next power of 2
	fftSize := 1
	for fftSize < n {
		fftSize *= 2
	}

	frame := make([]complex128, fftSize)
	win := window.Hann(n)
	for i := 0; i < n; i++ {
		frame[i] = complex(samples[i]*win[i], 0)
	}

	spectrum := fft.FFT(frame)
	mag := make([]float64, fftSize/2+1)
	for i := range mag {
		mag[i] = cmplx.Abs(spectrum[i])
	}

	return mag
}

// TimbreSpecToFloat64 converts a TimbreSpecBins array to a float64 slice for KNN.
func TimbreSpecToFloat64(spec [TimbreSpecBins]float64) []float64 {
	out := make([]float64, TimbreSpecBins)
	copy(out, spec[:])
	return out
}
