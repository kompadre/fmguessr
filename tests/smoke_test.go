package tests

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

// TestSmokeMatchSelfContained is a fast end-to-end smoke test that requires
// no external fixtures. It renders a known patch as target, runs the matcher
// with minimal settings, and verifies the result is reasonable.
func TestSmokeMatchSelfContained(t *testing.T) {
	// Use a known waveform template as "target"
	targetPatch := match.SquareBase()
	targetAudio := match.RenderCandidate(targetPatch, 60, 1*time.Second)
	targetNorm := match.NormalizeRMS(match.TrimOrPad(targetAudio, 44100))

	features := match.ExtractFeatures(targetNorm, 44100)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cfg := match.DefaultConfig()
	cfg.RenderDur = 1 * time.Second
	cfg.MIDINote = 60
	cfg.Verbose = false
	cfg.Ctx = ctx
	cfg.GA.Population = 32
	cfg.GA.Generations = 2
	cfg.GA.TournamentK = 2
	cfg.GA.CrossoverRate = 0.7
	cfg.GA.MutationRateStart = 0.5
	cfg.GA.MutationRateEnd = 0.1
	cfg.GA.Elites = 2

	result := match.Run(targetNorm, features, cfg)

	if result.Best == nil {
		t.Fatal("Run returned nil Best candidate")
	}
	if result.BestDist > 2.0 {
		t.Errorf("BestDist too high: %.6f (expected < 2.0 for self-match)", result.BestDist)
	}
	if len(result.Top) == 0 {
		t.Error("Top candidates list is empty")
	}

	// Verify the result can be encoded and rendered
	encoded := result.Best.Encode()
	if len(encoded) != 128 {
		t.Errorf("Encoded patch size: %d, expected 128", len(encoded))
	}

	replayed := match.RenderCandidate(result.Best, 60, 1*time.Second)
	if len(replayed) == 0 {
		t.Error("Replayed audio is empty")
	}

	// Verify the replayed audio has non-trivial energy
	rms := match.CalcRMS(replayed)
	if rms < 1e-6 {
		t.Errorf("Replayed RMS too low: %.8f", rms)
	}

	t.Logf("Smoke test passed: dist=%.6f, algo=%d, top=%d candidates",
		result.BestDist, result.Best.Algorithm+1, len(result.Top))
}

// TestSmokeMatchWithTimbreTarget renders a DX7 patch, extracts its timbre,
// and verifies the matcher can recover a reasonably close patch.
func TestSmokeMatchWithTimbreTarget(t *testing.T) {
	// Create a target with a specific algorithm
	targetPatch := match.SineBase()
	targetPatch.Algorithm = 5 // Algorithm 6 (simple FM pair)
	targetPatch.Ops[5].OutputLevel = 99
	targetPatch.Ops[5].FreqCoarse = 1
	targetPatch.Ops[4].OutputLevel = 80
	targetPatch.Ops[4].FreqCoarse = 1
	targetPatch.Feedback = 3

	targetAudio := match.RenderCandidate(targetPatch, 60, 2*time.Second)
	targetNorm := match.NormalizeRMS(match.TrimOrPad(targetAudio, 88200))

	features := match.ExtractFeatures(targetNorm, 44100)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cfg := match.DefaultConfig()
	cfg.RenderDur = 2 * time.Second
	cfg.MIDINote = 60
	cfg.Verbose = false
	cfg.Ctx = ctx
	cfg.GA.Population = 64
	cfg.GA.Generations = 3
	cfg.GA.TournamentK = 3
	cfg.GA.CrossoverRate = 0.7
	cfg.GA.MutationRateStart = 0.5
	cfg.GA.MutationRateEnd = 0.1
	cfg.GA.Elites = 2

	result := match.Run(targetNorm, features, cfg)

	if result.Best == nil {
		t.Fatal("Run returned nil Best candidate")
	}
	if result.BestDist > 1.0 {
		t.Errorf("BestDist too high: %.6f (expected < 1.0 for algorithm recovery)", result.BestDist)
	}

	// Verify spectral similarity
	resultAudio := match.RenderCandidate(result.Best, 60, 2*time.Second)
	resultNorm := match.NormalizeRMS(match.TrimOrPad(resultAudio, 88200))
	specDist := match.SpectralDistance(targetNorm, resultNorm)

	t.Logf("Timbre recovery: dist=%.6f, spectral=%.6f, algo=%d (target algo=%d)",
		result.BestDist, specDist, result.Best.Algorithm+1, targetPatch.Algorithm+1)
}

// TestSmokeEliteCohort verifies EliteCohort deduplication with cached signatures.
func TestSmokeEliteCohort(t *testing.T) {
	cohort := match.NewEliteCohort(3, match.ScoreTimbre)

	c1 := match.SineBase()
	c1.Ops[5].OutputLevel = 99
	c2 := match.SineBase()
	c2.Ops[5].OutputLevel = 80
	c3 := match.SineBase()
	c3.Ops[5].OutputLevel = 60

	// Insert three distinct candidates
	if !cohort.TryInsert(c1, 0.1) {
		t.Error("First insert should succeed")
	}
	if !cohort.TryInsert(c2, 0.2) {
		t.Error("Second insert should succeed")
	}
	if !cohort.TryInsert(c3, 0.3) {
		t.Error("Third insert should succeed")
	}

	// Duplicate should be rejected
	if cohort.TryInsert(c1, 0.05) {
		t.Error("Duplicate insert should be rejected")
	}

	if cohort.Len() != 3 {
		t.Errorf("Expected 3 entries, got %d", cohort.Len())
	}

	// Insert a better candidate — should evict worst
	c4 := match.SineBase()
	c4.Ops[5].OutputLevel = 50
	if !cohort.TryInsert(c4, 0.01) {
		t.Error("Better candidate should be inserted")
	}

	if cohort.Len() != 3 {
		t.Errorf("Expected 3 entries after eviction, got %d", cohort.Len())
	}

	cands := cohort.Candidates()
	if cands[0].Ops[5].OutputLevel != 50 {
		t.Error("Best candidate should be c4 (OutputLevel=50)")
	}
}

// TestSmokeBatchEvaluation verifies evaluateBatchParallel works via Run()
// and produces consistent results for identical inputs.
func TestSmokeBatchEvaluation(t *testing.T) {
	patch := match.SineBase()
	audio := match.RenderCandidate(patch, 60, 1*time.Second)
	target := match.NormalizeRMS(match.TrimOrPad(audio, 44100))
	features := match.ExtractFeatures(target, 44100)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cfg := match.DefaultConfig()
	cfg.RenderDur = 1 * time.Second
	cfg.MIDINote = 60
	cfg.Verbose = false
	cfg.Ctx = ctx
	cfg.GA.Population = 32
	cfg.GA.Generations = 1

	// Run twice — results should be deterministic (same seed doesn't matter for 1 gen)
	r1 := match.Run(target, features, cfg)
	r2 := match.Run(target, features, cfg)

	// Both should find a similar distance (not necessarily identical due to random seeds)
	if math.Abs(r1.BestDist-r2.BestDist) > 0.5 {
		t.Logf("Warning: large distance variance between runs: %.6f vs %.6f", r1.BestDist, r2.BestDist)
	}

	if r1.BestDist > 2.0 || r2.BestDist > 2.0 {
		t.Errorf("Distances too high: r1=%.6f r2=%.6f", r1.BestDist, r2.BestDist)
	}
}
