package match

import (
	"math"
	"math/cmplx"
	"sort"

	"github.com/madelynnblue/go-dsp/fft"
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
	win := getHannWindow(nFFT)

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

// ExtractTimbreWindowWithPeak extracts the timbre spectrum and returns the peak sample index.
// This avoids re-computing the peak when both the timbre spec and peak location are needed.
func ExtractTimbreWindowWithPeak(samples []float64, sampleRate int) ([TimbreSpecBins]float64, int) {
	spec := ExtractTimbreWindow(samples, sampleRate)

	// Find peak sample using the same 256-sample frame analysis
	const frameSize = 256
	nFrames := len(samples) / frameSize
	if nFrames == 0 {
		return spec, 0
	}

	peak := 0.0
	peakIdx := 0
	for i := 0; i < nFrames; i++ {
		sum := 0.0
		for j := 0; j < frameSize; j++ {
			s := samples[i*frameSize+j]
			sum += s * s
		}
		rms := math.Sqrt(sum / float64(frameSize))
		if rms > peak {
			peak = rms
			peakIdx = i
		}
	}

	return spec, peakIdx * frameSize
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
	win := getHannWindow(n)
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

const (
	TemporalNumSlices     = 4   // Attack, Decay, Sustain, Release phase slices
	TemporalSliceBins     = 128 // Bins per slice
	TemporalProfileFrames = 8   // Frames for RMS energy envelope trajectory
)

// TemporalTimbreSpec holds multi-stage spectral, flux, and envelope trajectory features.
type TemporalTimbreSpec struct {
	// StaticSpec is the 256-bin legacy summary (128 median + 128 variance)
	StaticSpec [TimbreSpecBins]float64

	// TemporalSlices holds 4 phase spectral slices (Attack, Decay, Sustain, Release)
	TemporalSlices [TemporalNumSlices][TemporalSliceBins]float64

	// EnergyProfile holds 8-frame normalized RMS energy trajectory over sound duration
	EnergyProfile [TemporalProfileFrames]float64

	// FluxTrajectory holds spectral flux rates between sequential temporal frames
	FluxTrajectory [TemporalProfileFrames - 1]float64
}

// ToLegacySpec returns the 256-bin static spectrum for backward compatibility with 1D predictors.
func (t TemporalTimbreSpec) ToLegacySpec() [TimbreSpecBins]float64 {
	return t.StaticSpec
}

// ToExtendedVector flattens all static, temporal slice, envelope, and flux features into a 1D vector.
func (t TemporalTimbreSpec) ToExtendedVector() []float64 {
	out := make([]float64, 0, TimbreSpecBins+TemporalNumSlices*TemporalSliceBins+TemporalProfileFrames+(TemporalProfileFrames-1))
	out = append(out, t.StaticSpec[:]...)
	for s := 0; s < TemporalNumSlices; s++ {
		out = append(out, t.TemporalSlices[s][:]...)
	}
	out = append(out, t.EnergyProfile[:]...)
	out = append(out, t.FluxTrajectory[:]...)
	return out
}

// ExtractTemporalTimbreSpec extracts static, temporal multi-slice, energy envelope, and flux features.
func ExtractTemporalTimbreSpec(samples []float64, sampleRate int) TemporalTimbreSpec {
	var spec TemporalTimbreSpec
	spec.StaticSpec = ExtractTimbreWindow(samples, sampleRate)

	if len(samples) < 1024 {
		return spec
	}

	// 1. Find active signal bounds
	maxPeak := 0.0
	for _, s := range samples {
		if abs := math.Abs(s); abs > maxPeak {
			maxPeak = abs
		}
	}
	if maxPeak < 1e-4 {
		return spec
	}

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
	if activeLen < 1024 {
		return spec
	}
	activeSamples := samples[activeStart:activeEnd]

	// 2. Compute 8-frame RMS energy profile
	frameLen := activeLen / TemporalProfileFrames
	if frameLen < 128 {
		frameLen = 128
	}

	var maxRMS float64
	for i := 0; i < TemporalProfileFrames; i++ {
		st := i * frameLen
		en := st + frameLen
		if en > len(activeSamples) {
			en = len(activeSamples)
		}
		if st >= en {
			break
		}

		sumSq := 0.0
		for _, v := range activeSamples[st:en] {
			sumSq += v * v
		}
		rms := math.Sqrt(sumSq / float64(en-st))
		spec.EnergyProfile[i] = rms
		if rms > maxRMS {
			maxRMS = rms
		}
	}

	// Normalize energy profile to [0, 1]
	if maxRMS > 1e-8 {
		for i := range spec.EnergyProfile {
			spec.EnergyProfile[i] /= maxRMS
		}
	}

	// 3. Compute 4 temporal phase slices (Attack, Decay, Sustain, Release)
	sliceRanges := [TemporalNumSlices][2]float64{
		{0.00, 0.15},
		{0.15, 0.40},
		{0.40, 0.85},
		{0.85, 1.00},
	}

	for s := 0; s < TemporalNumSlices; s++ {
		stIdx := int(float64(activeLen) * sliceRanges[s][0])
		enIdx := int(float64(activeLen) * sliceRanges[s][1])
		if enIdx <= stIdx+64 {
			enIdx = stIdx + 64
		}
		if enIdx > activeLen {
			enIdx = activeLen
		}
		if stIdx >= activeLen {
			stIdx = activeLen - 64
		}

		sliceSamples := activeSamples[stIdx:enIdx]
		mag := fftMag(sliceSamples)
		if mag == nil {
			continue
		}

		// Downsample/compress magnitude spectrum into 128 bins up to 8kHz
		halfWin := len(mag)
		maxBin := int(float64(halfWin) * 8000.0 / float64(sampleRate))
		if maxBin < TemporalSliceBins {
			maxBin = TemporalSliceBins
		}
		if maxBin > halfWin {
			maxBin = halfWin
		}

		binsPerOutput := maxBin / TemporalSliceBins
		if binsPerOutput < 1 {
			binsPerOutput = 1
		}

		sliceMax := 0.0
		for i := 0; i < TemporalSliceBins; i++ {
			sum := 0.0
			count := 0
			for j := i * binsPerOutput; j < (i+1)*binsPerOutput && j < maxBin; j++ {
				sum += mag[j]
				count++
			}
			if count > 0 {
				val := sum / float64(count)
				spec.TemporalSlices[s][i] = val
				if val > sliceMax {
					sliceMax = val
				}
			}
		}

		// Normalize slice to [0, 1]
		if sliceMax > 1e-8 {
			for i := range spec.TemporalSlices[s] {
				spec.TemporalSlices[s][i] /= sliceMax
			}
		}
	}

	// 4. Compute Spectral Flux Trajectory between sequential frames
	for i := 0; i < TemporalProfileFrames-1; i++ {
		dEnv := spec.EnergyProfile[i+1] - spec.EnergyProfile[i]
		spec.FluxTrajectory[i] = math.Abs(dEnv)
	}

	return spec
}

// TemporalTimbreDistance computes a weighted distance incorporating static spectrum,
// multi-phase temporal slices, energy envelope, and flux trajectories.
func TemporalTimbreDistance(a, b TemporalTimbreSpec) float64 {
	// 1. Static spectrum Euclidean distance (normalized)
	var staticSum float64
	for i := 0; i < TimbreSpecBins; i++ {
		d := a.StaticSpec[i] - b.StaticSpec[i]
		staticSum += d * d
	}
	staticDist := math.Sqrt(staticSum) / math.Sqrt(float64(TimbreSpecBins))

	// 2. Multi-phase temporal slices distance
	var slicesDist float64
	for s := 0; s < TemporalNumSlices; s++ {
		var sliceSum float64
		for i := 0; i < TemporalSliceBins; i++ {
			d := a.TemporalSlices[s][i] - b.TemporalSlices[s][i]
			sliceSum += d * d
		}
		slicesDist += math.Sqrt(sliceSum) / math.Sqrt(float64(TemporalSliceBins))
	}
	slicesDist /= float64(TemporalNumSlices)

	// 3. Energy profile RMS envelope distance
	var envSum float64
	for i := 0; i < TemporalProfileFrames; i++ {
		d := a.EnergyProfile[i] - b.EnergyProfile[i]
		envSum += d * d
	}
	envDist := math.Sqrt(envSum) / math.Sqrt(float64(TemporalProfileFrames))

	// 4. Flux trajectory distance
	var fluxSum float64
	for i := 0; i < TemporalProfileFrames-1; i++ {
		d := a.FluxTrajectory[i] - b.FluxTrajectory[i]
		fluxSum += d * d
	}
	fluxDist := math.Sqrt(fluxSum) / math.Sqrt(float64(TemporalProfileFrames-1))

	// Composite weighted distance
	return 0.40*staticDist + 0.30*slicesDist + 0.20*envDist + 0.10*fluxDist
}
