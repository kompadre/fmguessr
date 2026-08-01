package match_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestGateDurationDetectionAndRendering(t *testing.T) {
	// 1. Create target with 400ms sustained active sound and 600ms silence
	sampleRate := 44100
	activeLen := int(float64(sampleRate) * 0.4) // 400ms
	totalLen := 44100

	target := make([]float64, totalLen)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < activeLen; i++ {
		// Sustained signal
		target[i] = (rng.Float64()*2.0 - 1.0) * 0.5
	}

	// Extract features
	features := match.ExtractFeatures(target, sampleRate)

	t.Logf("Detected GateMs: %.2f ms (expected ~400ms)", features.GateMs)
	if features.GateMs < 350 || features.GateMs > 450 {
		t.Errorf("Expected GateMs ~400ms, got %.2f ms", features.GateMs)
	}

	// 2. Render candidate with gate duration
	cand := match.RandomCandidate(rng)
	gateDur := time.Duration(features.GateMs * float64(time.Millisecond))
	renderDur := 1 * time.Second

	rendered := match.RenderCandidateWithGate(cand, 60, gateDur, renderDur)
	if len(rendered) == 0 {
		t.Fatalf("RenderCandidateWithGate returned empty audio")
	}

	// Total length should include 1s renderDur + 250ms drain tail
	expectedSamples := 44100 + 44100/4
	if len(rendered) != expectedSamples {
		t.Errorf("Expected %d samples, got %d", expectedSamples, len(rendered))
	}
}
