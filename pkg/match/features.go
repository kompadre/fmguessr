package match

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/madelynnblue/go-dsp/fft"
)

// TargetFeatures holds acoustic features extracted from the target audio.
type TargetFeatures struct {
	// Pitch
	F0                 float64 // Fundamental frequency in Hz
	F0Confidence       float64 // 0.0-1.0
	PitchJumpSemitones float64 // Semitone pitch jump detected during attack (e.g. +4 for chiptune arpeggio)

	// Envelope
	TargetRMS        float64 // Un-normalized RMS energy of target audio
	AttackMs         float64 // Attack time in ms
	AttackPeakSample int     // Sample index where peak amplitude occurs
	DecayMs          float64 // Decay time in ms
	SustainLevel     float64 // 0.0-1.0
	ReleaseMs        float64 // Release time in ms
	GateMs           float64 // Key hold gate duration in ms

	// Spectral
	SpectralCentroidHz float64
	BrightnessScore    float64 // 0.0 (warm) to 1.0 (bright)
	Harmonicity        float64 // 0.0 (inharmonic) to 1.0 (harmonic)

	// Dominant frequency peaks (relative to F0)
	FrequencyRatios []float64
}

// ExtractBrightness computes spectral features focusing on attack transients (first 450ms).
func ExtractBrightness(samples []float64, sampleRate int) TargetFeatures {
	f := TargetFeatures{}
	if len(samples) < 512 {
		return f
	}

	// Limit scanning to the first 20,000 samples (~450ms) where plucks/strikes occur
	scanLimit := 20000
	if len(samples) < scanLimit {
		scanLimit = len(samples)
	}

	frameSize := 1024
	hopSize := 512
	maxBrightness := 0.0
	maxCentroid := 0.0

	for start := 0; start+frameSize <= scanLimit; start += hopSize {
		frame := samples[start : start+frameSize]
		mag := computeSpectrum(frame, sampleRate)
		if mag == nil {
			continue
		}

		brightness := computeBrightness(mag, sampleRate)
		centroid := spectralCentroid(mag, sampleRate)

		// RMS to weight frames - only consider frames with signal
		sum := 0.0
		for _, s := range frame {
			sum += s * s
		}
		rms := math.Sqrt(sum / float64(frameSize))

		if rms > 0.005 {
			if brightness > maxBrightness {
				maxBrightness = brightness
			}
			if centroid > maxCentroid {
				maxCentroid = centroid
			}
		}
	}

	f.BrightnessScore = maxBrightness
	f.SpectralCentroidHz = maxCentroid
	return f
}

func ExtractFeatures(samples []float64, sampleRate int) TargetFeatures {
	f := TargetFeatures{}
	f.TargetRMS = CalcRMS(samples)

	// Pitch detection & jump trajectory
	f.F0, f.F0Confidence = detectPitch(samples, sampleRate)
	f.PitchJumpSemitones = detectPitchJump(samples, sampleRate)

	// Envelope extraction
	f.AttackMs, f.AttackPeakSample, f.DecayMs, f.SustainLevel, f.ReleaseMs, f.GateMs = estimateEnvelope(samples, sampleRate)

	// Spectral features: scan the whole sample to find maximum brightness/centroid
	bright := ExtractBrightness(samples, sampleRate)
	f.BrightnessScore = bright.BrightnessScore
	f.SpectralCentroidHz = bright.SpectralCentroidHz

	// For harmonicity and peaks, we still use a steady-state portion
	ssStart := len(samples) / 4
	ssEnd := len(samples) * 3 / 4
	if ssEnd-ssStart < 2048 {
		ssStart = 0
		ssEnd = len(samples)
	}
	ss := samples[ssStart:ssEnd]
	spec := computeSpectrum(ss, sampleRate)

	f.Harmonicity = estimateHarmonicity(f.F0, spec, sampleRate)
	peaks := findPeaks(spec, sampleRate)
	f.FrequencyRatios = estimateFrequencyRatios(f.F0, peaks)

	return f
}

// HeuristicCandidate generates a smart initial candidate from target features.
// If noteFreq > 0, it overrides the detected F0 (user-specified note).
func HeuristicCandidate(features TargetFeatures, noteFreq float64, rng interface{ Intn(int) int }) *Candidate {
	// Override F0 if user specified a note frequency
	if noteFreq > 0 {
		features.F0 = noteFreq
		features.F0Confidence = 1.0
		// Recompute frequency ratios with the known F0
		if len(features.FrequencyRatios) == 0 || features.FrequencyRatios[0] != 1.0 {
			// Re-derive ratios: assume first peak is near F0, rest are sidebands
			// Keep existing ratios but ensure ratio 1.0 is present
			hasOne := false
			for _, r := range features.FrequencyRatios {
				if r == 1.0 {
					hasOne = true
					break
				}
			}
			if !hasOne {
				features.FrequencyRatios = append([]float64{1.0}, features.FrequencyRatios...)
			}
		}
	}

	c := &Candidate{Scores: nanScores()}
	for i := 0; i < 26; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}

	// Algorithm selection based on harmonicity and brightness
	c.Algorithm = selectAlgorithm(features)

	// Feedback: moderate for harmonic, higher for inharmonic
	if features.Harmonicity > 0.7 {
		c.Feedback = 2
	} else {
		c.Feedback = 4
	}

	// Operator parameters based on frequency ratios and brightness
	c.Ops = estimateOperatorParams(c.Algorithm, features)

	c.PitchRate1 = 99
	c.PitchLevel1 = 50
	c.PitchLevel2 = 50
	if math.Abs(features.PitchJumpSemitones) >= 1.0 {
		c.PitchRate1 = 82 // ~50ms transition
		c.PitchLevel1 = 50
		lvl2 := int(50 + math.Round(features.PitchJumpSemitones))
		if lvl2 < 0 {
			lvl2 = 0
		}
		if lvl2 > 99 {
			lvl2 = 99
		}
		c.PitchLevel2 = uint8(lvl2)
	}

	return c
}

// HeuristicCandidateWithAlgo is like HeuristicCandidate but forces a specific algorithm.
func HeuristicCandidateWithAlgo(features TargetFeatures, noteFreq float64, algo uint8) *Candidate {
	// Override F0 if user specified a note frequency
	if noteFreq > 0 {
		features.F0 = noteFreq
		features.F0Confidence = 1.0
		// Recompute frequency ratios with the known F0
		if len(features.FrequencyRatios) == 0 || features.FrequencyRatios[0] != 1.0 {
			hasOne := false
			for _, r := range features.FrequencyRatios {
				if r == 1.0 {
					hasOne = true
					break
				}
			}
			if !hasOne {
				features.FrequencyRatios = append([]float64{1.0}, features.FrequencyRatios...)
			}
		}
	}

	c := &Candidate{Scores: nanScores()}
	for i := 0; i < 26; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}

	c.Algorithm = algo

	// Feedback: moderate for harmonic, higher for inharmonic
	if features.Harmonicity > 0.7 {
		c.Feedback = 2
	} else {
		c.Feedback = 4
	}

	c.Ops = estimateOperatorParams(c.Algorithm, features)

	c.PitchRate1 = 99
	c.PitchLevel1 = 50
	c.PitchLevel2 = 50
	if math.Abs(features.PitchJumpSemitones) >= 1.0 {
		c.PitchRate1 = 82 // ~50ms transition
		c.PitchLevel1 = 50
		lvl2 := int(50 + math.Round(features.PitchJumpSemitones))
		if lvl2 < 0 {
			lvl2 = 0
		}
		if lvl2 > 99 {
			lvl2 = 99
		}
		c.PitchLevel2 = uint8(lvl2)
	}

	return c
}

// --- Pitch Detection (Autocorrelation) ---

func detectPitchFrame(frame []float64, sampleRate int) (float64, float64) {
	if len(frame) < 1024 {
		return 0, 0
	}
	frameSize := len(frame)
	buf := make([]float64, frameSize)
	mean := 0.0
	for _, s := range frame {
		mean += s
	}
	mean /= float64(frameSize)
	for i, s := range frame {
		buf[i] = s - mean
	}

	win := getHannWindow(frameSize)
	for i := range buf {
		buf[i] *= win[i]
	}

	fftIn := make([]complex128, frameSize)
	for i := range buf {
		fftIn[i] = complex(buf[i], 0)
	}
	spectrum := fft.FFT(fftIn)
	power := make([]complex128, len(spectrum))
	for i, v := range spectrum {
		power[i] = v * cmplx.Conj(v)
	}
	acorr := fft.IFFT(power)
	acorrReal := make([]float64, len(acorr))
	for i, v := range acorr {
		acorrReal[i] = real(v)
	}

	minLag := sampleRate / 2000
	maxLag := sampleRate / 50
	if minLag < 2 {
		minLag = 2
	}
	if maxLag > len(acorrReal)/2 {
		maxLag = len(acorrReal) / 2
	}

	type peak struct {
		lag int
		val float64
	}
	var peaks []peak
	absMax := 0.0
	for lag := minLag; lag < maxLag; lag++ {
		if acorrReal[lag] > acorrReal[lag-1] && acorrReal[lag] > acorrReal[lag+1] {
			peaks = append(peaks, peak{lag, acorrReal[lag]})
			if acorrReal[lag] > absMax {
				absMax = acorrReal[lag]
			}
		}
	}

	if absMax < 1e-10 {
		return 0, 0
	}

	bestLag := 0
	threshold := absMax * 0.70
	// Iterate from longest lag to shortest (fundamental first, highest harmonics last).
	// For periodic signals like foghorns, the fundamental at the longest lag is the true F0.
	for i := len(peaks) - 1; i >= 0; i-- {
		if peaks[i].val >= threshold {
			bestLag = peaks[i].lag
			break
		}
	}

	if bestLag == 0 {
		return 0, 0
	}

	// Octave correction: when detected frequency is very low (< 80 Hz),
	// it's likely a sub-harmonic. Check if a higher harmonic (shorter lag)
	// at 1/2, 1/3, 1/4 of the lag is also a strong peak.
	f0 := float64(sampleRate) / float64(bestLag)
	if f0 < 80.0 {
		for _, div := range []int{2, 3, 4} {
			candidateLag := bestLag / div
			if candidateLag >= minLag && acorrReal[candidateLag] >= threshold {
				bestLag = candidateLag
			}
		}
	}

	conf := acorrReal[bestLag] / acorrReal[0]
	f0 = float64(sampleRate) / float64(bestLag)
	return f0, math.Min(conf, 1.0)
}

func detectPitch(samples []float64, sampleRate int) (float64, float64) {
	if len(samples) < 2048 {
		return 0, 0
	}

	frameSize := 2048
	hopSize := 1024

	type detection struct {
		f0   float64
		conf float64
		rms  float64
	}

	var all []detection

	for start := 0; start+frameSize <= len(samples); start += hopSize {
		frame := samples[start : start+frameSize]

		// Compute frame RMS for energy weighting
		var sumSq float64
		for _, s := range frame {
			sumSq += s * s
		}
		rms := math.Sqrt(sumSq / float64(frameSize))

		f0, conf := detectPitchFrame(frame, sampleRate)
		if f0 > 20 && conf > 0.2 {
			all = append(all, detection{f0, conf, rms})
		}
	}

	if len(all) == 0 {
		return 0, 0
	}

	// Normalize energies so attack transients don't dominate
	maxRMS := 0.0
	for _, d := range all {
		if d.rms > maxRMS {
			maxRMS = d.rms
		}
	}

	// Group detections by frequency (±10% tolerance)
	type freqGroup struct {
		freq     float64
		count    int
		bestConf float64
		totalW   float64 // weighted sum for weighted mean
	}

	var groups []freqGroup
	for _, d := range all {
		// Energy weight: quiet frames get downweighted (prevents attack skew)
		energyW := 0.3 + 0.7*(d.rms/maxRMS) // range [0.3, 1.0]
		score := d.conf * energyW

		found := false
		for g := range groups {
			if math.Abs(d.f0-groups[g].freq)/groups[g].freq < 0.10 {
				groups[g].count++
				groups[g].totalW += score
				if d.conf > groups[g].bestConf {
					groups[g].bestConf = d.conf
				}
				// Update running weighted-mean frequency
				groups[g].freq = (groups[g].freq*float64(groups[g].count-1) + d.f0) / float64(groups[g].count)
				found = true
				break
			}
		}
		if !found {
			groups = append(groups, freqGroup{freq: d.f0, count: 1, bestConf: d.conf, totalW: score})
		}
	}

	// Sort groups by weighted score (count × total confidence × energy weight)
	bestGroup := &groups[0]
	for g := range groups {
		if groups[g].totalW > bestGroup.totalW {
			bestGroup = &groups[g]
		}
	}

	// Sub-harmonic correction: if the winning frequency is very low,
	// check if a harmonic at 2× or 3× also has a strong group
	if bestGroup.freq < 80.0 {
		bestIdx := -1
		for i := range groups {
			if &groups[i] == bestGroup {
				bestIdx = i
				break
			}
		}
		for _, mult := range []int{2, 3} {
			for g := range groups {
				if g != bestIdx && math.Abs(groups[g].freq-bestGroup.freq*float64(mult))/bestGroup.freq < 0.15 {
					// Higher harmonic exists with reasonable support
					if groups[g].totalW > bestGroup.totalW*0.6 {
						bestGroup = &groups[g]
					}
				}
			}
		}
	}

	return bestGroup.freq, math.Min(bestGroup.bestConf, 1.0)
}

func detectPitchJump(samples []float64, sampleRate int) float64 {
	if len(samples) < 3072 {
		return 0
	}
	f0Start, conf0 := detectPitchFrame(samples[0:2048], sampleRate)
	f0Jump, conf1 := detectPitchFrame(samples[1024:3072], sampleRate)
	if conf0 > 0.5 && conf1 > 0.5 && f0Start > 0 && f0Jump > 0 {
		ratio := f0Jump / f0Start
		semitones := 12.0 * math.Log2(ratio)
		if math.Abs(semitones) >= 1.0 && math.Abs(semitones) <= 24.0 {
			return math.Round(semitones)
		}
	}
	return 0
}

// --- Envelope Extraction ---

func estimateEnvelope(samples []float64, sampleRate int) (attackMs float64, peakSample int, decayMs, sustainLevel, releaseMs, gateMs float64) {
	if len(samples) == 0 {
		return
	}

	// Compute RMS in small frames
	frameSize := 256
	nFrames := len(samples) / frameSize
	if nFrames == 0 {
		return
	}

	rms := make([]float64, nFrames)
	for i := 0; i < nFrames; i++ {
		sum := 0.0
		for j := 0; j < frameSize; j++ {
			s := samples[i*frameSize+j]
			sum += s * s
		}
		rms[i] = math.Sqrt(sum / float64(frameSize))
	}

	// Find peak RMS
	peak := 0.0
	peakIdx := 0
	for i, v := range rms {
		if v > peak {
			peak = v
			peakIdx = i
		}
	}

	peakSample = peakIdx * frameSize

	if peak < 1e-10 {
		return
	}

	// Attack: time from first non-negligible sample to peak
	attackThreshold := peak * 0.1
	attackStart := 0
	for i, v := range rms {
		if v > attackThreshold {
			attackStart = i
			break
		}
	}
	attackMs = float64(peakIdx-attackStart) * float64(frameSize) / float64(sampleRate) * 1000.0
	if attackMs < 0 {
		attackMs = 0
	}

	// Sound End: find last active frame > 5% peak amplitude
	soundEndThreshold := peak * 0.05
	soundEndIdx := nFrames - 1
	for i := nFrames - 1; i >= 0; i-- {
		if rms[i] > soundEndThreshold {
			soundEndIdx = i
			break
		}
	}

	// Sustain level: measure steady-state RMS in middle section of active sound (20% to 70% of active range)
	sustainStart := peakIdx + int(float64(soundEndIdx-peakIdx)*0.20)
	sustainEnd := peakIdx + int(float64(soundEndIdx-peakIdx)*0.70)
	if sustainStart < peakIdx+1 {
		sustainStart = peakIdx + 1
	}
	if sustainEnd <= sustainStart {
		sustainEnd = sustainStart + 1
	}

	sustainSum := 0.0
	sustainCount := 0
	for i := sustainStart; i < sustainEnd && i < len(rms); i++ {
		sustainSum += rms[i]
		sustainCount++
	}
	sustainRMS := 0.0
	if sustainCount > 0 {
		sustainRMS = sustainSum / float64(sustainCount)
		sustainLevel = sustainRMS / peak
	}

	// Tail level: measure RMS in the final 20% of the active sound before soundEndIdx
	tailStart := peakIdx + int(float64(soundEndIdx-peakIdx)*0.80)
	if tailStart < peakIdx {
		tailStart = peakIdx
	}
	tailSum := 0.0
	tailCount := 0
	for i := tailStart; i <= soundEndIdx && i < len(rms); i++ {
		tailSum += rms[i]
		tailCount++
	}
	tailLevel := 0.0
	if tailCount > 0 {
		tailLevel = (tailSum / float64(tailCount)) / peak
	}

	// Key release / note-off detection (releaseStartIdx):
	// For sustained sounds (tailLevel >= 0.12 and sustainLevel >= 0.20), detect where the signal drops below 70% of sustainRMS.
	releaseStartIdx := soundEndIdx
	if tailLevel >= 0.12 && sustainLevel >= 0.20 {
		for i := sustainEnd; i <= soundEndIdx && i < len(rms); i++ {
			if rms[i] < sustainRMS*0.70 {
				releaseStartIdx = i
				break
			}
		}
	} else {
		// Decaying / percussive sound: set sustainLevel to tailLevel
		sustainLevel = tailLevel
	}

	// Decay: time from peak to where signal reaches sustain level
	sustainThreshold := peak * sustainLevel * 1.1
	decayEnd := peakIdx
	for i := peakIdx; i < nFrames; i++ {
		if rms[i] <= sustainThreshold {
			decayEnd = i
			break
		}
	}
	decayMs = float64(decayEnd-peakIdx) * float64(frameSize) / float64(sampleRate) * 1000.0
	if decayMs < 0 {
		decayMs = 0
	}

	// Key hold gate duration: time from initial attack to key release (note-off)
	gateFrames := releaseStartIdx - attackStart
	if gateFrames < 1 {
		gateFrames = 1
	}
	gateMs = float64(gateFrames) * float64(frameSize) / float64(sampleRate) * 1000.0

	// Release time: time from key release (note-off) to sound end (< 5% peak)
	relFrames := soundEndIdx - releaseStartIdx
	if relFrames < 0 {
		relFrames = 0
	}
	releaseMs = float64(relFrames) * float64(frameSize) / float64(sampleRate) * 1000.0

	// For percussive/one-shot decaying sounds (tailLevel < 0.12 or sustainLevel < 0.15), cap key hold gate to 50ms drum trigger
	if tailLevel < 0.12 || sustainLevel < 0.15 {
		if gateMs > 50.0 {
			gateMs = 50.0
		}
	}

	return
}

// computeSpectrum calculates the magnitude spectrum of a signal frame.
func computeSpectrum(samples []float64, sampleRate int) []float64 {
	n := len(samples)
	if n < 64 {
		return nil
	}

	// Pad to next power of 2 for FFT
	fftSize := 1
	for fftSize < n {
		fftSize *= 2
	}

	frame := make([]complex128, fftSize)
	// Apply Hann window ONLY to the actual signal portion, then pad
	win := getHannWindow(n)
	for i := 0; i < n; i++ {
		frame[i] = complex(samples[i]*win[i], 0)
	}

	spectrum := fft.FFT(frame)

	// Take magnitude of positive frequencies (N/2 + 1 bins)
	mag := make([]float64, fftSize/2+1)
	for i := range mag {
		mag[i] = cmplx.Abs(spectrum[i])
	}

	return mag
}

func spectralCentroid(mag []float64, sampleRate int) float64 {
	numerator := 0.0
	denominator := 0.0
	n := len(mag)
	if n <= 1 {
		return 0
	}
	// fftSize is (n-1)*2
	fftSize := (n - 1) * 2
	for i := 1; i < n; i++ {
		freq := float64(i) * float64(sampleRate) / float64(fftSize)
		numerator += freq * mag[i]
		denominator += mag[i]
	}
	if denominator < 1e-10 {
		return 0
	}
	return numerator / denominator
}

func computeBrightness(mag []float64, sampleRate int) float64 {
	n := len(mag)
	if n <= 1 {
		return 0
	}
	fftSize := (n - 1) * 2

	// High-frequency energy ratio (above ~2000 Hz for high-mid brightness)
	totalEnergy := 0.0
	hfEnergy := 0.0
	cutoffBin := int(2000.0 * float64(fftSize) / float64(sampleRate))
	if cutoffBin >= n {
		cutoffBin = n - 1
	}

	for i := 1; i < n; i++ {
		e := mag[i] * mag[i]
		totalEnergy += e
		if i >= cutoffBin {
			hfEnergy += e
		}
	}

	if totalEnergy < 1e-10 {
		return 0
	}

	ratio := hfEnergy / totalEnergy
	// Normalize: boost the ratio. For 2kHz+ energy, 20% is extremely bright, so scale by 5.
	brightness := math.Min(ratio*5, 1.0)
	return brightness
}

func estimateHarmonicity(f0 float64, mag []float64, sampleRate int) float64 {
	if f0 < 20 {
		return 0
	}

	n := len(mag)
	binWidth := float64(sampleRate) / float64(2*n)

	harmonicEnergy := 0.0
	totalEnergy := 0.0

	for i := 1; i < n; i++ {
		freq := float64(i) * binWidth
		e := mag[i] * mag[i]
		totalEnergy += e

		// Check if this bin is near a harmonic of F0
		harmonicNum := freq / f0
		nearestHarmonic := math.Round(harmonicNum)
		if nearestHarmonic < 1 {
			nearestHarmonic = 1
		}
		harmonicFreq := f0 * nearestHarmonic
		binDist := math.Abs(freq-harmonicFreq) / binWidth

		if binDist < 2.0 { // within 2 bins of a harmonic
			harmonicEnergy += e
		}
	}

	if totalEnergy < 1e-10 {
		return 0
	}

	return harmonicEnergy / totalEnergy
}

// --- Peak Picking ---

type spectralPeak struct {
	freqHz   float64
	amplitude float64
}

func findPeaks(mag []float64, sampleRate int) []spectralPeak {
	n := len(mag)
	if n < 3 {
		return nil
	}

	// Find noise floor (median of magnitudes)
	sorted := make([]float64, n)
	copy(sorted, mag)
	for i := range sorted {
		if sorted[i] < 0 {
			sorted[i] = -sorted[i]
		}
	}
	// Simple sort for median
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	noiseFloor := sorted[n/2]

	// Threshold: noise floor + 20dB worth of linear amplitude
	threshold := noiseFloor * 10.0 // ~20dB above median

	var peaks []spectralPeak
	for i := 2; i < n-1; i++ {
		if mag[i] > mag[i-1] && mag[i] > mag[i+1] && mag[i] > threshold {
			freq := float64(i) * float64(sampleRate) / float64(2*n)
			peaks = append(peaks, spectralPeak{freqHz: freq, amplitude: mag[i]})
		}
	}

	// Sort by amplitude (strongest first)
	for i := 0; i < len(peaks)-1; i++ {
		for j := i + 1; j < len(peaks); j++ {
			if peaks[j].amplitude > peaks[i].amplitude {
				peaks[i], peaks[j] = peaks[j], peaks[i]
			}
		}
	}

	// Limit to top 10 peaks
	if len(peaks) > 10 {
		peaks = peaks[:10]
	}

	return peaks
}

func estimateFrequencyRatios(f0 float64, peaks []spectralPeak) []float64 {
	if f0 < 20 || len(peaks) == 0 {
		return []float64{1.0}
	}

	var ratios []float64
	for _, p := range peaks {
		ratio := p.freqHz / f0
		// Round to nearest 0.5 step for cleaner ratios
		ratio = math.Round(ratio*2) / 2
		if ratio < 0.5 {
			ratio = 0.5
		}
		if ratio > 31 {
			ratio = 31
		}
		ratios = append(ratios, ratio)
	}

	// Ensure we have at least ratio 1.0
	hasOne := false
	for _, r := range ratios {
		if math.Abs(r-1.0) < 0.3 {
			hasOne = true
			break
		}
	}
	if !hasOne {
		ratios = append([]float64{1.0}, ratios...)
	}

	return ratios
}

// --- Heuristic Parameter Estimation ---

func selectAlgorithm(f TargetFeatures) uint8 {
	// Decision tree based on features
	if f.F0Confidence < 0.3 && f.BrightnessScore > 0.5 {
		// Unpitched / noisy → additive or feedback-heavy
		return 31 // Algo 32 (all carriers, additive-like)
	}

	if f.Harmonicity > 0.8 {
		// Strongly harmonic → simple topologies with 2-3 carriers
		if f.AttackMs < 10 && f.DecayMs > 500 {
			// Percussive harmonic → bell-like algo
			return 0 // Algo 1 (deep serial chain)
		}
		if f.BrightnessScore > 0.6 {
			// Bright harmonic
			return 11 // Algo 12 (3 carriers, 3 modulators)
		}
		return 13 // Algo 14 (simple, 2 carriers)
	}

	if f.Harmonicity < 0.5 {
		// Inharmonic → deep serial chains
		if f.BrightnessScore > 0.6 {
			return 3 // Algo 4 (deep chain, 2 carriers)
		}
		return 0 // Algo 1 (deepest chain)
	}

	// Mixed → moderate complexity
	if f.AttackMs < 20 {
		// Fast attack → bell/pluck
		return 4 // Algo 5 (epiano-style)
	}

	return 11 // Algo 12 (versatile)
}

func estimateOperatorParams(algo uint8, f TargetFeatures) [6]OpParams {
	var ops [6]OpParams

	ratios := f.FrequencyRatios
	if len(ratios) == 0 {
		ratios = []float64{1.0}
	}

	for i := 0; i < 6; i++ {
		// Assign frequency ratio from peaks
		var ratio float64
		if i < len(ratios) {
			ratio = ratios[i]
		} else {
			// Cycle through available ratios
			ratio = ratios[i%len(ratios)]
		}

		// Convert ratio to coarse+fine
		coarse := uint8(ratio)
		fine := uint8((ratio - float64(coarse)) * 100)
		if coarse > 31 {
			coarse = 31
		}
		if fine > 99 {
			fine = 99
		}

		ops[i].Mode = 0 // ratio mode
		ops[i].FreqCoarse = coarse
		ops[i].FreqFine = fine

		// Output level based on brightness and carrier/modulator role
		// DX7 scale: 99 is max volume (0dB), 80 is -12dB, <70 is very quiet.
		if IsCarrier(algo, i) {
			// Carriers -> full volume for audible output (95-99)
			ops[i].OutputLevel = 95 + uint8(f.BrightnessScore*4)
		} else {
			// Modulators -> level scales with brightness (75-99)
			ops[i].OutputLevel = 75 + uint8(f.BrightnessScore*24)
		}

		// Envelope estimation from target features (ADSR)
		// Rate1 (Attack): faster attack = higher rate
		ops[i].Rate1 = TimeMsToDX7Rate(f.AttackMs)

		// Rate2 (Decay): faster decay = higher rate
		r2 := TimeMsToDX7Rate(f.DecayMs)
		if !IsCarrier(algo, i) {
			// Modulators decay slightly faster for punchy attack transients
			if r2 < 90 {
				r2 += 8
			}
		}
		ops[i].Rate2 = r2

		// Level3 (Sustain Level): map linear 0..1 to DX7 0.75dB scale (0..99)
		l3 := AmplitudeToDX7Level(f.SustainLevel)
		if f.SustainLevel < 0.20 {
			l3 = 0 // Percussive/one-shot sounds decay to silence during Note On
		} else if IsCarrier(algo, i) && l3 < 60 {
			l3 = 60 // Sustained sounds maintain audible carrier level
		}
		ops[i].Level3 = l3

		// Rate4 (Release): for percussive/one-shot sounds, release rate must be fast (85-99)
		r4 := TimeMsToDX7Rate(f.ReleaseMs)
		if f.SustainLevel < 0.20 {
			if r4 < 85 {
				r4 = 85 // Fast release for drum/percussive hits
			}
		}
		ops[i].Rate4 = r4
	}

	return ops
}

// AmplitudeToDX7Level converts linear amplitude (0.0 - 1.0) to DX7 0.75dB level scale (0 - 99).
func AmplitudeToDX7Level(amp float64) uint8 {
	if amp <= 0.001 {
		return 0
	}
	if amp >= 1.0 {
		return 99
	}
	db := 20.0 * math.Log10(amp)
	level := 99.0 + db/0.75
	if level < 0 {
		return 0
	}
	if level > 99 {
		return 99
	}
	return uint8(math.Round(level))
}

// TimeMsToDX7Rate converts time duration in ms to DX7 EG Rate (0 - 99).
func TimeMsToDX7Rate(ms float64) uint8 {
	if ms <= 1.0 {
		return 99
	}
	if ms >= 30000.0 {
		return 10
	}
	rate := 99.0 - 15.0*math.Log10(ms/10.0)
	if rate < 10 {
		return 10
	}
	if rate > 99 {
		return 99
	}
	return uint8(math.Round(rate))
}

// MidiNoteName converts a MIDI note number (0-127) to its musical string representation (e.g., "C3").
func MidiNoteName(note int) string {
	names := []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
	octave := note/12 - 1
	name := names[note%12]
	return fmt.Sprintf("%s%d", name, octave)
}
