package match

import (
	"math"
)

const (
	formantLifterCoeffs = 18   // cepstral lifter cutoff
	formantFFTSize      = 2048 // FFT size for formant extraction (21.5 Hz resolution at 44.1kHz)
	formantMaxPeaks     = 6    // maximum formants to extract
)

// FormantPeak represents a spectral resonance (formant or FM sideband peak).
type FormantPeak struct {
	Frequency float64 // Center frequency in Hz
	Amplitude float64 // Relative amplitude (0-1)
}

// ExtractSpectralEnvelope computes the smoothed spectral envelope from
// the log-magnitude spectrum using a moving average smoother.
// Returns linear-magnitude envelope for positive frequency bins.
func ExtractSpectralEnvelope(samples []float64, sampleRate int) []float64 {
	if len(samples) < 64 {
		return nil
	}

	// Use sustain region (skip first 100ms, use up to 2s)
	skipSamples := sampleRate / 10 // 100ms
	if skipSamples >= len(samples) {
		skipSamples = 0
	}
	region := samples[skipSamples:]
	if len(region) > sampleRate*2 { // cap at 2s
		region = region[:sampleRate*2]
	}

	// Use proven computeSpectrum from features.go
	mag := computeSpectrum(region, sampleRate)
	if len(mag) == 0 {
		return nil
	}

	halfBins := len(mag)

	// Log magnitude
	logMag := make([]float64, halfBins)
	for i := range mag {
		if mag[i] < 1e-10 {
			logMag[i] = -23.0 // log(1e-10)
		} else {
			logMag[i] = math.Log(mag[i])
		}
	}

	// Smooth: moving average (bandwidth ~430 Hz at 43 Hz/bin resolution)
	smoothRadius := 10
	smoothed := make([]float64, halfBins)
	for i := 0; i < halfBins; i++ {
		sum := 0.0
		count := 0
		for j := i - smoothRadius; j <= i+smoothRadius; j++ {
			if j >= 0 && j < halfBins {
				sum += logMag[j]
				count++
			}
		}
		smoothed[i] = sum / float64(count)
	}

	// Exponentiate to linear magnitude
	envelope := make([]float64, halfBins)
	for i := range smoothed {
		envelope[i] = math.Exp(smoothed[i])
	}

	return envelope
}

// ExtractFormants extracts the N dominant peaks from the spectral envelope.
// Returns formants ordered by frequency (F1 < F2 < F3 ...).
func ExtractFormants(samples []float64, sampleRate int, nFormants int) []FormantPeak {
	if nFormants <= 0 {
		nFormants = formantMaxPeaks
	}

	envelope := ExtractSpectralEnvelope(samples, sampleRate)
	if len(envelope) == 0 {
		return nil
	}

	// Compute the FFT size that computeSpectrum would use
	n := len(samples)
	skipSamples := sampleRate / 10
	if skipSamples >= n {
		skipSamples = 0
	}
	region := n - skipSamples
	if region > sampleRate*2 {
		region = sampleRate * 2
	}
	fftSize := 1
	for fftSize < region {
		fftSize *= 2
	}

	return findEnvelopePeaks(envelope, sampleRate, fftSize, nFormants)
}

// findEnvelopePeaks finds peaks in a smoothed spectral envelope.
// Returns up to maxPeaks, ordered by frequency.
func findEnvelopePeaks(mag []float64, sampleRate, fftSize, maxPeaks int) []FormantPeak {
	n := len(mag)
	if n < 3 {
		return nil
	}

	// Noise floor (median)
	sorted := make([]float64, n)
	copy(sorted, mag)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	noiseFloor := sorted[n/2]
	threshold := noiseFloor * 3.0

	// Find local maxima
	type peak struct {
		bin int
		amp float64
	}
	var rawPeaks []peak
	for i := 2; i < n-2; i++ {
		if mag[i] > mag[i-1] && mag[i] > mag[i-2] &&
			mag[i] > mag[i+1] && mag[i] > mag[i+2] &&
			mag[i] > threshold {
			rawPeaks = append(rawPeaks, peak{i, mag[i]})
		}
	}

	// Sort by amplitude descending
	for i := 0; i < len(rawPeaks); i++ {
		for j := i + 1; j < len(rawPeaks); j++ {
			if rawPeaks[j].amp > rawPeaks[i].amp {
				rawPeaks[i], rawPeaks[j] = rawPeaks[j], rawPeaks[i]
			}
		}
	}
	if len(rawPeaks) > maxPeaks {
		rawPeaks = rawPeaks[:maxPeaks]
	}

	// Normalize amplitude
	maxAmp := 0.0
	for _, p := range rawPeaks {
		if p.amp > maxAmp {
			maxAmp = p.amp
		}
	}

	result := make([]FormantPeak, len(rawPeaks))
	for i, p := range rawPeaks {
		freqHz := float64(p.bin) * float64(sampleRate) / float64(fftSize)
		ampNorm := 0.0
		if maxAmp > 0 {
			ampNorm = p.amp / maxAmp
		}
		result[i] = FormantPeak{Frequency: freqHz, Amplitude: ampNorm}
	}

	// Sort by frequency
	for i := 0; i < len(result); i++ {
		for j := i + 1; j < len(result); j++ {
			if result[j].Frequency < result[i].Frequency {
				result[i], result[j] = result[j], result[i]
			}
		}
	}

	return result
}

// ExtractFormantTrajectory extracts formants at each STFT frame.
func ExtractFormantTrajectory(samples []float64, sampleRate, hopSize, nFormants int) [][]FormantPeak {
	if nFormants <= 0 {
		nFormants = formantMaxPeaks
	}
	winSize := formantFFTSize
	if winSize > len(samples) {
		winSize = nextPowerOf2(len(samples))
	}

	nFrames := (len(samples)-winSize)/hopSize + 1
	if nFrames < 1 {
		nFrames = 1
	}

	win := getHannWindow(winSize)
	result := make([][]FormantPeak, nFrames)

	for i := 0; i < nFrames; i++ {
		start := i * hopSize
		end := start + winSize
		if end > len(samples) {
			end = len(samples)
		}
		frame := make([]float64, winSize)
		n := copy(frame, samples[start:end])
		for j := 0; j < n; j++ {
			frame[j] *= win[j]
		}
		result[i] = ExtractFormants(frame, sampleRate, nFormants)
	}
	return result
}

// formantString returns a human-readable formant summary for logging.
func formantString(peaks []FormantPeak) string {
	if len(peaks) == 0 {
		return "none"
	}
	s := ""
	for i, p := range peaks {
		if i > 0 {
			s += ", "
		}
		s += formatHz(p.Frequency)
	}
	return s
}

// formatHz formats a frequency as a short string.
func formatHz(hz float64) string {
	if hz >= 1000 {
		return formatFloat(hz/1000, 1) + "kHz"
	}
	return formatFloat(hz, 0) + "Hz"
}

func formatFloat(f float64, decimals int) string {
	if decimals == 0 {
		return formatInt(int(math.Round(f)))
	}
	return formatInt(int(f))
}

func formatInt(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	if n < 0 {
		s = "-"
		n = -n
	}
	digits := ""
	for n > 0 {
		digits = string(rune('0'+n%10)) + digits
		n /= 10
	}
	return s + digits
}
