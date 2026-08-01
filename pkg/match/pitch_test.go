package match_test

import (
	"os"
	"testing"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestChiptuneCoinWavPitchAndPitchEG(t *testing.T) {
	// Look for coin.wav in workspace root (../../coin.wav or ./coin.wav)
	paths := []string{"../../coin.wav", "coin.wav", "../coin.wav"}
	var coinPath string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			coinPath = p
			break
		}
	}

	if coinPath == "" {
		t.Skip("coin.wav not found in test search paths, skipping integration test")
	}

	samples, sampleRate, err := match.LoadWav(coinPath)
	if err != nil {
		t.Fatalf("Failed to load coin.wav: %v", err)
	}

	features := match.ExtractFeatures(samples, sampleRate)

	// Fundamental pitch should be ~1470 Hz (F#6 / MIDI 90), NOT 50.5 Hz (G#1 / MIDI 32)
	t.Logf("Detected F0: %.1f Hz (confidence: %.2f)", features.F0, features.F0Confidence)
	if features.F0 < 1000 || features.F0 > 1800 {
		t.Errorf("Expected fundamental F0 ~1470 Hz (F#6 / MIDI 90), got %.1f Hz", features.F0)
	}

	// Pitch jump should be ~ +4 semitones (F#6 -> A#6)
	t.Logf("Detected PitchJumpSemitones: %.1f", features.PitchJumpSemitones)
	if features.PitchJumpSemitones < 3.0 || features.PitchJumpSemitones > 6.0 {
		t.Errorf("Expected PitchJumpSemitones ~ +4.0 semitones, got %.1f", features.PitchJumpSemitones)
	}

	cand := match.HeuristicCandidate(features, 0, nil)
	t.Logf("Heuristic PitchEG: Rate1=%d, Level1=%d, Level2=%d", cand.PitchRate1, cand.PitchLevel1, cand.PitchLevel2)
	if cand.PitchLevel2 < 53 || cand.PitchLevel2 > 56 {
		t.Errorf("Expected PitchLevel2 ~ 54 (+4 semitones), got %d", cand.PitchLevel2)
	}
}
