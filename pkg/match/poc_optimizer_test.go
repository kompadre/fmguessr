package match_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestPOCOptimizerExactMatch(t *testing.T) {
	rng := rand.New(rand.NewSource(1234))
	cand := match.RandomCandidate(rng)

	renderDur := 1 * time.Second
	target := match.RenderCandidate(cand, 60, renderDur)

	pocOpt := match.NewPOCOptimizer(target, cand, 5, true, 60, renderDur)
	refined := pocOpt.Run()

	t.Logf("POCOptimizer starting/final distance for exact match: %.6f", pocOpt.BestDist())
	if pocOpt.BestDist() > 1e-4 {
		t.Errorf("POCOptimizer exact match mismatch: expected <= 0.0001, got %.6f", pocOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("POCOptimizer returned nil patch")
	}
}

func TestPOCOptimizerMultiKnobPerturbation(t *testing.T) {
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

	pocOpt := match.NewPOCOptimizer(target, perturbed, 5, true, 60, renderDur)
	refined := pocOpt.Run()

	t.Logf("Refined dist after POCOptimizer: %.6f", pocOpt.BestDist())

	if pocOpt.BestDist() >= initialDist {
		t.Errorf("POCOptimizer failed to improve perturbed candidate distance (initial: %.6f, final: %.6f)", initialDist, pocOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("POCOptimizer returned nil patch")
	}
}
