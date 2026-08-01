package match_test

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestDiffOptimizerExactMatch(t *testing.T) {
	rng := rand.New(rand.NewSource(1234))
	cand := match.RandomCandidate(rng)

	renderDur := 1 * time.Second
	target := match.RenderCandidate(cand, 60, renderDur)

	diffOpt := match.NewDiffOptimizer(target, cand, 5, true, 60, renderDur)
	refined := diffOpt.Run()

	t.Logf("DiffOptimizer starting/final distance for exact match: %.6f", diffOpt.BestDist())
	if diffOpt.BestDist() > 1e-4 {
		t.Errorf("DiffOptimizer exact match mismatch: expected <= 0.0001, got %.6f", diffOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("DiffOptimizer returned nil patch")
	}
}

func TestDiffOptimizerMultiKnobPerturbation(t *testing.T) {
	rng := rand.New(rand.NewSource(5678))
	cand := match.RandomCandidate(rng)
	cand.Algorithm = 0 // Algo 1

	renderDur := 1 * time.Second
	target := match.RenderCandidate(cand, 60, renderDur)

	// Perturb multiple coupled knobs simultaneously
	perturbed := cand.Copy()
	perturbed.Ops[0].OutputLevel = uint8(max(0, int(cand.Ops[0].OutputLevel)-10))
	perturbed.Ops[1].OutputLevel = uint8(max(0, int(cand.Ops[1].OutputLevel)-8))
	perturbed.Ops[1].FreqFine = uint8(min(99, int(cand.Ops[1].FreqFine)+15))

	initialDist := match.SpectralDistance(match.NormalizeRMS(target), match.NormalizeRMS(match.RenderCandidate(perturbed, 60, renderDur)))
	t.Logf("Initial perturbed dist: %.6f", initialDist)

	diffOpt := match.NewDiffOptimizer(target, perturbed, 5, true, 60, renderDur)
	refined := diffOpt.Run()

	t.Logf("Refined dist after DiffOptimizer: %.6f", diffOpt.BestDist())

	if diffOpt.BestDist() >= initialDist {
		t.Errorf("DiffOptimizer failed to improve perturbed candidate distance (initial: %.6f, final: %.6f)", initialDist, diffOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("DiffOptimizer returned nil patch")
	}
}

func TestDiffOptimizerCrunchArbitraryWav(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "arbitrary_target.wav")

	// Render a synthesized target WAV to simulate an arbitrary WAV file input
	rng := rand.New(rand.NewSource(9999))
	targetCand := match.RandomCandidate(rng)
	targetCand.Algorithm = 0

	samples := match.RenderCandidate(targetCand, 60, 1*time.Second)
	if err := match.SaveWav(wavPath, samples, 44100); err != nil {
		t.Fatalf("Failed to write target WAV file: %v", err)
	}

	// Create a starting seed patch (slightly perturbed from target)
	seed := targetCand.Copy()
	seed.Ops[0].OutputLevel = uint8(max(0, int(targetCand.Ops[0].OutputLevel)-6))
	seed.Ops[1].FreqFine = uint8(min(99, int(targetCand.Ops[1].FreqFine)+5))

	initialDist := match.SpectralDistance(match.NormalizeRMS(samples), match.NormalizeRMS(match.RenderCandidate(seed, 60, 1*time.Second)))
	t.Logf("Initial CrunchWavFile seed dist: %.6f", initialDist)

	bestPatch, finalDist, err := match.CrunchWavFile(wavPath, seed, 5, true)
	if err != nil {
		t.Fatalf("CrunchWavFile failed: %v", err)
	}

	t.Logf("CrunchWavFile completed! Initial: %.6f -> Final Spectral Distance: %.6f", initialDist, finalDist)
	if bestPatch == nil {
		t.Fatalf("CrunchWavFile returned nil patch")
	}
	if finalDist >= initialDist {
		t.Errorf("CrunchWavFile failed to reduce distance: initial %.6f, final %.6f", initialDist, finalDist)
	}

	// Clean up temp file
	_ = os.Remove(wavPath)
}

func TestDiffOptimizerAlternatingPasses(t *testing.T) {
	rng := rand.New(rand.NewSource(4321))
	targetCand := match.RandomCandidate(rng)
	targetCand.Algorithm = 4

	renderDur := 1 * time.Second
	targetPcm := match.RenderCandidate(targetCand, 60, renderDur)

	// Perturb both timbre AND temporal parameters
	seed := targetCand.Copy()
	seed.Ops[0].OutputLevel = uint8(max(0, int(targetCand.Ops[0].OutputLevel)-12))
	seed.Ops[1].FreqFine = uint8(min(99, int(targetCand.Ops[1].FreqFine)+10))
	seed.Ops[0].Rate1 = uint8(max(0, int(targetCand.Ops[0].Rate1)-25)) // Slow attack
	seed.Ops[0].Level3 = uint8(max(0, int(targetCand.Ops[0].Level3)-30)) // Lower sustain

	initDist := match.SpectralDistance(match.NormalizeRMS(targetPcm), match.NormalizeRMS(match.RenderCandidate(seed, 60, renderDur)))
	t.Logf("Initial dual-perturbed distance: %.6f", initDist)

	diffOpt := match.NewDiffOptimizer(targetPcm, seed, 5, true, 60, renderDur)
	refined := diffOpt.RunAlternating(3)

	finalDist := diffOpt.BestDist()
	t.Logf("Final distance after 3 alternating cycles: %.6f (Reduction: %.6f)", finalDist, initDist-finalDist)

	if refined == nil {
		t.Fatalf("RunAlternating returned nil patch")
	}
	if finalDist >= initDist {
		t.Errorf("RunAlternating failed to reduce distance: initial %.6f, final %.6f", initDist, finalDist)
	}
}
