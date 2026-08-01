package match_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestKeyWavEnvelopeExtraction(t *testing.T) {
	keyPath := filepath.Join("..", "..", "key.wav")
	samples, sampleRate, err := match.LoadWav(keyPath)
	if err != nil {
		t.Skipf("key.wav not found at %s: %v", keyPath, err)
	}

	features := match.ExtractFeatures(samples, sampleRate)

	t.Logf("key.wav extracted features: GateMs=%.1fms, ReleaseMs=%.1fms, SustainLevel=%.2f",
		features.GateMs, features.ReleaseMs, features.SustainLevel)

	if math.Abs(features.GateMs-1000.0) > 150.0 {
		t.Errorf("Expected GateMs ~1000ms for key.wav, got %.1fms", features.GateMs)
	}

	if features.ReleaseMs < 200.0 {
		t.Errorf("Expected ReleaseMs >= 200ms for key.wav release decay, got %.1fms", features.ReleaseMs)
	}

	if features.SustainLevel < 0.20 {
		t.Errorf("Expected SustainLevel >= 0.20 for sustained key sound, got %.2f", features.SustainLevel)
	}
}
