package match

import (
	"math"
	"testing"
	"time"
)

func TestTemporalTimbreSpecExtraction(t *testing.T) {
	// Generate dummy audio: 1 second sine wave at 440 Hz with decaying envelope
	sampleRate := 44100
	nSamples := sampleRate
	samples := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		sec := float64(i) / float64(sampleRate)
		env := math.Exp(-2.0 * sec) // Decaying envelope
		samples[i] = env * math.Sin(2.0*math.Pi*440.0*sec)
	}

	spec := ExtractTemporalTimbreSpec(samples, sampleRate)

	// 1. Check backward-compatible legacy static spec
	legacySpec := spec.ToLegacySpec()
	if len(legacySpec) != TimbreSpecBins {
		t.Fatalf("Expected legacy spec length %d, got %d", TimbreSpecBins, len(legacySpec))
	}

	// 2. Check extended vector flattening length
	extVec := spec.ToExtendedVector()
	expectedLen := TimbreSpecBins + TemporalNumSlices*TemporalSliceBins + TemporalProfileFrames + (TemporalProfileFrames - 1)
	if len(extVec) != expectedLen {
		t.Fatalf("Expected extended vector length %d, got %d", expectedLen, len(extVec))
	}

	// 3. Verify Energy Profile shows decay
	if spec.EnergyProfile[0] <= spec.EnergyProfile[TemporalProfileFrames-1] {
		t.Errorf("Expected energy profile decay, but t_0 (%.4f) <= t_last (%.4f)",
			spec.EnergyProfile[0], spec.EnergyProfile[TemporalProfileFrames-1])
	}

	// 4. Test self-distance is 0
	distSelf := TemporalTimbreDistance(spec, spec)
	if math.Abs(distSelf) > 1e-6 {
		t.Errorf("Expected zero self distance, got %f", distSelf)
	}
}

func TestTemporalSpectralDistance(t *testing.T) {
	// Synthesize candidate patch
	cand1 := DecodeVoicePatch(epiano)
	pcm1 := RenderCandidate(cand1, 60, 1*time.Second)

	// Modified patch with higher feedback
	cand2 := cand1.Copy()
	cand2.Feedback = 7
	pcm2 := RenderCandidate(cand2, 60, 1*time.Second)

	distSame := TemporalSpectralDistance(pcm1, pcm1, 44100)
	distDiff := TemporalSpectralDistance(pcm1, pcm2, 44100)

	if distSame != 0.0 {
		t.Errorf("Expected 0.0 distance for identical PCM, got %f", distSame)
	}
	if distDiff <= 0.0 {
		t.Errorf("Expected positive distance for different patches, got %f", distDiff)
	}
}
