package match_test

import (
	"context"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestParseNoteStringOrNumber(t *testing.T) {
	tests := []struct {
		input    string
		expected uint8
	}{
		{"C3", 48},
		{"c3", 48},
		{"C4", 60},
		{"60", 60},
		{"A4", 69},
		{"C#3", 49},
		{"Db3", 49},
	}

	for _, tt := range tests {
		note, _, err := match.ParseNoteStringOrNumber(tt.input)
		if err != nil {
			t.Errorf("ParseNoteStringOrNumber(%s) returned error: %v", tt.input, err)
		}
		if note != tt.expected {
			t.Errorf("ParseNoteStringOrNumber(%s) = %d, expected %d", tt.input, note, tt.expected)
		}
	}
}

func TestApplyFeatureOverrides(t *testing.T) {
	orig := match.TargetFeatures{
		F0:           220.0,
		F0Confidence: 0.9,
		AttackMs:     10.0,
		DecayMs:      100.0,
		SustainLevel: 0.5,
		ReleaseMs:    50.0,
		GateMs:       200.0,
	}

	atk := 5.0
	gate := 150.0
	ov := match.FeatureOverrides{
		Note:     "A4",
		AttackMs: &atk,
		GateMs:   &gate,
	}

	res, midiNote := match.ApplyFeatureOverrides(orig, ov)

	if midiNote != 69 {
		t.Errorf("Expected MIDI note 69 (A4), got %d", midiNote)
	}
	if res.AttackMs != 5.0 {
		t.Errorf("Expected AttackMs 5.0, got %.1f", res.AttackMs)
	}
	if res.GateMs != 150.0 {
		t.Errorf("Expected GateMs 150.0, got %.1f", res.GateMs)
	}
	if res.DecayMs != 100.0 {
		t.Errorf("DecayMs should be unchanged 100.0, got %.1f", res.DecayMs)
	}
}

func TestProbeAudioWithoutCodebook(t *testing.T) {
	sampleRate := 44100
	rng := rand.New(rand.NewSource(101))
	cand := match.RandomCandidate(rng)
	knownAudio := match.RenderCandidate(cand, 60, 1*time.Second)

	// Probe without codebook — should complete fast and return features
	res := match.ProbeAudio(knownAudio, sampleRate, match.FeatureOverrides{Note: "60"})

	if res.MIDINote != 60 {
		t.Errorf("Expected MIDINote=60, got %d", res.MIDINote)
	}
	if res.F0Hz <= 0 {
		t.Errorf("Expected positive F0Hz, got %f", res.F0Hz)
	}
	// ProbeResult should not have SeenBefore or BestMatch fields anymore
}

func TestStep1CodebookEarlyReturn(t *testing.T) {
	tempDir := t.TempDir()
	cbPath := filepath.Join(tempDir, "early_return_codebook.bin")

	rng := rand.New(rand.NewSource(202))
	cand := match.RandomCandidate(rng)
	knownAudio := match.RenderCandidate(cand, 60, 1*time.Second)

	err := match.IndexCandidateToCodebook(cand, "EARLY_RETURN_PATCH", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("Failed to index candidate: %v", err)
	}

	features := match.ExtractFeatures(knownAudio, 44100)
	var midiNote int
	features, midiNote = match.ApplyFeatureOverrides(features, match.FeatureOverrides{Note: "60"})

	gaCfg := match.DefaultGAConfig()
	gaCfg.Generations = 100 // Should NOT run 100 generations because Step 1 returns early!

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := match.Config{
		RenderDur:    1 * time.Second,
		MIDINote:     uint8(midiNote),
		CodebookPath: cbPath,
		GA:           gaCfg,
		Ctx:          ctx,
	}

	start := time.Now()
	targetNorm := match.NormalizeRMS(knownAudio)
	result := match.Run(targetNorm, features, cfg)
	elapsed := time.Since(start)

	if result.BestDist > 1e-4 {
		t.Errorf("Expected early return match distance <= 0.0001, got %.6f", result.BestDist)
	}
	if elapsed > 2*time.Second {
		t.Errorf("Step 1 early return took too long: %v (should be < 0.5s)", elapsed)
	}
}
