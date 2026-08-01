package match_test

import (
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

func TestProbeAudioAndSeenBefore(t *testing.T) {
	tempDir := t.TempDir()
	cbPath := filepath.Join(tempDir, "probe_codebook.bin")

	// Render a known patch
	sampleRate := 44100
	rng := rand.New(rand.NewSource(101))
	cand := match.RandomCandidate(rng)
	knownAudio := match.RenderCandidate(cand, 60, 1*time.Second)

	// Index patch in codebook
	err := match.IndexCandidateToCodebook(cand, "KNOWN_PATCH", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("Failed to index candidate: %v", err)
	}

	// Probe the exact audio with matching root note
	res := match.ProbeAudio(knownAudio, sampleRate, cbPath, match.FeatureOverrides{Note: "60"})

	if !res.SeenBefore {
		t.Errorf("Expected SeenBefore = true for exact codebook audio, got false")
	}
	if res.BestMatch == nil {
		t.Fatalf("Expected BestMatch details to be non-nil")
	}
	if res.BestMatch.Name != "KNOWN_PATCH" {
		t.Errorf("Expected BestMatch.Name KNOWN_PATCH, got %s", res.BestMatch.Name)
	}
	if res.BestMatch.Distance > 1e-3 {
		t.Errorf("Expected exact match distance <= 0.001, got %.6f", res.BestMatch.Distance)
	}
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

	cfg := match.Config{
		RenderDur:    1 * time.Second,
		MIDINote:     uint8(midiNote),
		CodebookPath: cbPath,
		GA:           gaCfg,
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
