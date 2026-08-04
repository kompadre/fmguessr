package match

import "math"

// PeakShape captures the geometric envelope of a sound's energy peak.
// Used for ultra-fast metadata-based filtering of codebook entries
// before any rendering occurs.
type PeakShape struct {
	AttackTimeMs   float64 // Time from 10% to 90% peak energy (ms)
	TransientWidth float64 // Duration energy stays within -3dB of peak (ms)
	DecaySlope     float64 // RMS drop rate in first 50ms after peak (normalized/ms)
	CrestFactor    float64 // Peak-to-RMS ratio
}

// ExtractPeakShapeFromAudio analyzes raw PCM audio to extract its peak shape.
// Uses 256-sample frame RMS analysis (same resolution as estimateEnvelope).
func ExtractPeakShapeFromAudio(samples []float64, sampleRate int) PeakShape {
	var ps PeakShape
	if len(samples) < 512 {
		return ps
	}

	const frameSize = 256
	nFrames := len(samples) / frameSize
	if nFrames < 2 {
		return ps
	}

	// Compute per-frame RMS
	rms := make([]float64, nFrames)
	for i := 0; i < nFrames; i++ {
		sum := 0.0
		for j := 0; j < frameSize; j++ {
			s := samples[i*frameSize+j]
			sum += s * s
		}
		rms[i] = math.Sqrt(sum / float64(frameSize))
	}

	// Find peak frame
	peak := 0.0
	peakIdx := 0
	for i, v := range rms {
		if v > peak {
			peak = v
			peakIdx = i
		}
	}
	if peak < 1e-10 {
		return ps
	}

	// Attack time: 10% to 90% of peak
	attackThreshLo := peak * 0.1
	attackThreshHi := peak * 0.9
	attackStart := 0
	attackEnd := peakIdx
	for i := 0; i <= peakIdx; i++ {
		if rms[i] >= attackThreshLo && attackStart == 0 {
			attackStart = i
		}
		if rms[i] >= attackThreshHi {
			attackEnd = i
			break
		}
	}
	ps.AttackTimeMs = float64(attackEnd-attackStart) * float64(frameSize) / float64(sampleRate) * 1000.0
	if ps.AttackTimeMs < 0 {
		ps.AttackTimeMs = 0
	}

	// Transient width: frames within -3dB of peak (peak * 0.707)
	peakThresh3dB := peak * 0.707
	transientStart := peakIdx
	transientEnd := peakIdx
	for i := peakIdx; i >= 0; i-- {
		if rms[i] < peakThresh3dB {
			transientStart = i
			break
		}
	}
	for i := peakIdx; i < nFrames; i++ {
		if rms[i] < peakThresh3dB {
			transientEnd = i
			break
		}
	}
	ps.TransientWidth = float64(transientEnd-transientStart) * float64(frameSize) / float64(sampleRate) * 1000.0
	if ps.TransientWidth < 0 {
		ps.TransientWidth = 0
	}

	// Decay slope: RMS drop in first 50ms after peak
	decayFrames := int(0.050 * float64(sampleRate) / float64(frameSize))
	if decayFrames < 1 {
		decayFrames = 1
	}
	decayEnd := peakIdx + decayFrames
	if decayEnd >= nFrames {
		decayEnd = nFrames - 1
	}
	if decayEnd > peakIdx {
		ps.DecaySlope = (peak - rms[decayEnd]) / (float64(decayEnd-peakIdx) * float64(frameSize) / float64(sampleRate) * 1000.0)
	}
	if ps.DecaySlope < 0 {
		ps.DecaySlope = 0
	}

	// Crest factor: peak RMS / global RMS
	sumSq := 0.0
	for _, v := range rms {
		sumSq += v * v
	}
	globalRMS := math.Sqrt(sumSq / float64(nFrames))
	if globalRMS > 1e-10 {
		ps.CrestFactor = peak / globalRMS
	}

	return ps
}

// EstimateEnvelopeTime converts a DX7 envelope rate and level delta to approximate milliseconds.
// DX7 EG timing: Rate 99 ≈ fastest (~1ms per level unit), Rate 0 ≈ slowest (~8s per level unit).
// The relationship is roughly exponential: time_ms ≈ 8000 * 2^(-(rate-99)*3/28) * levelDelta/99
func EstimateEnvelopeTime(rate uint8, levelDelta uint8) float64 {
	if levelDelta == 0 {
		return 0
	}
	// Normalize rate to 0-1 scale (99 = fastest = 1.0, 0 = slowest ≈ 0.0)
	r := float64(rate) / 99.0
	// Exponential timing: at rate 99, ~0.5ms per level unit; at rate 0, ~8000ms per level unit
	// Base time per level unit: 0.5ms * 2^((1-r)*14) gives smooth range
	baseMsPerLevel := 0.5 * math.Pow(2, (1.0-r)*14.0)
	return baseMsPerLevel * float64(levelDelta)
}

// CompositePeakShape computes the effective peak shape of a DX7 patch by analyzing
// the carrier operators' envelope parameters. Carriers are summed in the output,
// so the composite envelope is the level-weighted sum of carrier envelopes.
func CompositePeakShape(algo uint8, ops [6]OpParams) PeakShape {
	var ps PeakShape
	if algo >= 32 {
		return ps
	}

	type carrierInfo struct {
		attackMs    float64
		decaySlope  float64
		outputLevel float64
		level3      float64
	}

	var carriers []carrierInfo
	totalLevel := 0.0

	for i := 0; i < 6; i++ {
		if !IsCarrier(algo, i) {
			continue
		}
		op := ops[i]
		ol := float64(op.OutputLevel)
		if ol < 1 {
			ol = 1
		}
		totalLevel += ol

		// Attack time: R1 drives level 0→L1 (L1=99 in DX7)
		attackMs := EstimateEnvelopeTime(op.Rate1, 99)

		// Decay slope: R2 drives L1→L2. We approximate L2 ≈ L3 (sustain level)
		// since Encode() sets L2 = L3 when no template override.
		levelDelta := 99 - int(op.Level3)
		if levelDelta < 0 {
			levelDelta = 0
		}
		decayTimeMs := EstimateEnvelopeTime(op.Rate2, uint8(levelDelta))
		decaySlope := 0.0
		if decayTimeMs > 0 {
			decaySlope = float64(levelDelta) / decayTimeMs // level units per ms
		}

		carriers = append(carriers, carrierInfo{
			attackMs:    attackMs,
			decaySlope:  decaySlope,
			outputLevel: ol,
			level3:      float64(op.Level3),
		})
	}

	if len(carriers) == 0 || totalLevel < 1 {
		// Fallback: use first operator as if it were a carrier
		if len(ops) > 0 {
			op := ops[0]
			ps.AttackTimeMs = EstimateEnvelopeTime(op.Rate1, 99)
			levelDelta := 99 - int(op.Level3)
			if levelDelta < 0 {
				levelDelta = 0
			}
			decayTimeMs := EstimateEnvelopeTime(op.Rate2, uint8(levelDelta))
			if decayTimeMs > 0 {
				ps.DecaySlope = float64(levelDelta) / decayTimeMs
			}
			ps.CrestFactor = 1.0
		}
		return ps
	}

	// Level-weighted composite attack time (louder carriers dominate)
	var weightedAttack float64
	var weightedDecay float64
	for _, c := range carriers {
		w := c.outputLevel / totalLevel
		weightedAttack += c.attackMs * w
		weightedDecay += c.decaySlope * w
	}
	ps.AttackTimeMs = weightedAttack
	ps.DecaySlope = weightedDecay

	// Transient width: max carrier transient (broader if multiple carriers with different attacks)
	maxTransient := 0.0
	for _, c := range carriers {
		// Approximate: transient ≈ attack time * 0.5
		t := c.attackMs * 0.5
		if t > maxTransient {
			maxTransient = t
		}
	}
	ps.TransientWidth = maxTransient

	// Crest factor: weighted average of level-based crest factor
	// Higher sustain level = lower crest factor (more sustained = less peaky)
	var weightedCrest float64
	for _, c := range carriers {
		sustainRatio := c.level3 / 99.0
		// Crest factor approximation: 1.0 (pure sustain) to 3.0 (sharp peak)
		cf := 1.0 + 2.0*(1.0-sustainRatio)
		weightedCrest += cf * (c.outputLevel / totalLevel)
	}
	ps.CrestFactor = weightedCrest

	return ps
}

// PeakShapeDistance computes a weighted Euclidean distance between two peak shapes.
// Lower values indicate more similar envelope characteristics.
func PeakShapeDistance(a, b PeakShape) float64 {
	// Normalize each component to comparable scales before weighting
	dAttack := (a.AttackTimeMs - b.AttackTimeMs) / 50.0   // 50ms = 1.0 unit
	dTransient := (a.TransientWidth - b.TransientWidth) / 30.0 // 30ms = 1.0 unit
	dDecay := (a.DecaySlope - b.DecaySlope) * 100.0        // scale up small slopes
	dCrest := (a.CrestFactor - b.CrestFactor) / 2.0        // 2.0 = 1.0 unit

	// Weights: attack time most important for percussive matching
	return 0.40*dAttack*dAttack +
		0.20*dTransient*dTransient +
		0.30*dDecay*dDecay +
		0.10*dCrest*dCrest
}
