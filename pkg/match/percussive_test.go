package match_test

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestPercussiveShortWavProcessing(t *testing.T) {
	// 1. Generate a synthetic 50ms hi-hat (high frequency noise with exponential decay)
	sampleRate := 44100
	hihatLen := int(float64(sampleRate) * 0.05) // 50ms
	totalLen := 44100                            // 1s total (50ms audio + 950ms silence)

	target := make([]float64, totalLen)
	rng := rand.New(rand.NewSource(12345))

	for i := 0; i < hihatLen; i++ {
		// White noise multiplied by exponential decay envelope
		decay := float64(hihatLen-i) / float64(hihatLen)
		target[i] = (rng.Float64()*2.0 - 1.0) * decay
	}

	// 2. Test ExtractTimbreWindow on short hi-hat
	timbreSpec := match.ExtractTimbreWindow(target, sampleRate)
	specSum := 0.0
	for _, v := range timbreSpec {
		specSum += v
	}
	if specSum <= 1e-5 {
		t.Fatalf("ExtractTimbreWindow returned near-zero spectrum for hi-hat audio: sum=%.6f", specSum)
	}

	// 3. Test NormalizeRMS on padded short hi-hat
	normTarget := match.NormalizeRMS(target)
	activeSum := 0.0
	for i := 0; i < hihatLen; i++ {
		activeSum += normTarget[i] * normTarget[i]
	}
	activeRMS := (activeSum / float64(hihatLen))
	if activeRMS < 1e-4 {
		t.Fatalf("NormalizeRMS failed to preserve active hi-hat signal level: RMS=%.6f", activeRMS)
	}

	// 4. Test SpectralDistance favors non-silent percussive candidate over silent candidate
	silentCandidate := make([]float64, totalLen)
	percussiveCandidate := make([]float64, totalLen)
	copy(percussiveCandidate, normTarget)

	distSilent := match.SpectralDistance(normTarget, silentCandidate)
	distPercussive := match.SpectralDistance(normTarget, percussiveCandidate)

	t.Logf("SpectralDistance to silent candidate: %.6f", distSilent)
	t.Logf("SpectralDistance to matching percussive candidate: %.6f", distPercussive)

	if distPercussive >= distSilent {
		t.Errorf("SpectralDistance failed: percussive match (%.6f) should be significantly lower than silent candidate (%.6f)",
			distPercussive, distSilent)
	}

	// 5. Test estimateOperatorParams allows Level3=0 for percussive features
	features := match.TargetFeatures{
		F0:              0,
		F0Confidence:    0,
		AttackMs:        2,
		DecayMs:         50,
		SustainLevel:    0.02, // Very low sustain
		ReleaseMs:       10,
		BrightnessScore: 0.8,
		Harmonicity:     0.1,
	}

	candidate := match.HeuristicCandidateWithAlgo(features, 0, 31) // Algo 32 (all carriers)
	carrierFound := false
	for opIdx := 0; opIdx < 6; opIdx++ {
		if match.IsCarrier(31, opIdx) {
			carrierFound = true
			if candidate.Ops[opIdx].Level3 != 0 {
				t.Errorf("Carrier op %d Level3 = %d, expected 0 for percussive sound",
					opIdx, candidate.Ops[opIdx].Level3)
			}
		}
	}
	if !carrierFound {
		t.Errorf("No carrier found for Algo 31")
	}
}

func TestKickDrumDecayAndRelease(t *testing.T) {
	// Generate a synthetic kick drum (sine sweep 120Hz -> 40Hz with 150ms exponential decay)
	sampleRate := 44100
	totalLen := 44100 // 1 sec
	target := make([]float64, totalLen)

	for i := 0; i < totalLen; i++ {
		sec := float64(i) / float64(sampleRate)
		if sec > 0.25 {
			break // audio stops after 250ms
		}
		// Pitch sweep 120Hz -> 40Hz
		freq := 120.0 - 80.0*(sec/0.25)
		phase := 2.0 * math.Pi * freq * sec
		decay := (0.25 - sec) / 0.25
		target[i] = math.Sin(phase) * 0.8 * decay * decay
	}

	features := match.ExtractFeatures(target, sampleRate)

	if features.GateMs > 60 {
		t.Errorf("Expected percussive kick GateMs <= 60ms, got %.2fms", features.GateMs)
	}

	cand := match.HeuristicCandidate(features, 0, rand.New(rand.NewSource(42)))
	gateDur := time.Duration(features.GateMs * float64(time.Millisecond))
	renderDur := 1 * time.Second

	rendered := match.RenderCandidateWithGate(cand, 36, gateDur, renderDur) // MIDI 36 = C1 kick note

	// Check audio volume at 350ms (0.35s = 15435 samples). It MUST be near zero (no sustained drone).
	tailSample := int(0.35 * float64(sampleRate))
	tailRMS := 0.0
	for i := tailSample; i < tailSample+1000 && i < len(rendered); i++ {
		tailRMS += rendered[i] * rendered[i]
	}
	tailRMS = (tailRMS / 1000.0)

	t.Logf("Kick rendered tail RMS at 350ms: %.6f (must be < 0.01)", tailRMS)
	if tailRMS > 0.01 {
		t.Errorf("Kick drum failed to release/decay cleanly: tail RMS at 350ms is %.6f", tailRMS)
	}
}
