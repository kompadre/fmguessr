package match_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestLiveOptimizerPreservesCandidateDistance(t *testing.T) {
	// Generate target audio
	rng := rand.New(rand.NewSource(1234))
	cand := match.RandomCandidate(rng)

	// Render target at 2.0s duration
	renderDur := 2 * time.Second
	target := match.RenderCandidate(cand, 60, renderDur)

	// Create LiveOptimizer with renderDur = 2.0s
	liveOpt := match.NewLiveOptimizer(target, cand, 10, false, 60, renderDur)
	refined := liveOpt.Run()

	// Since target was rendered directly from cand, liveOpt starting distance MUST be 0.000000
	t.Logf("LiveOptimizer starting/final distance at 2.0s: %.6f", liveOpt.BestDist())
	if liveOpt.BestDist() > 1e-4 {
		t.Errorf("LiveOptimizer distance mismatch: expected <= 0.0001 for exact candidate, got %.6f", liveOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("LiveOptimizer returned nil patch")
	}
}

func TestMultiKnobLiveOptimizerImprovement(t *testing.T) {
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

	liveOpt := match.NewLiveOptimizer(target, perturbed, 10, true, 60, renderDur)
	refined := liveOpt.Run()

	t.Logf("Initial perturbed dist: %.6f, Refined dist: %.6f", match.SpectralDistance(match.NormalizeRMS(target), match.NormalizeRMS(match.RenderCandidate(perturbed, 60, renderDur))), liveOpt.BestDist())

	if liveOpt.BestDist() >= 0.50 {
		t.Errorf("Multi-knob LiveOptimizer failed to optimize perturbed candidate, dist: %.6f", liveOpt.BestDist())
	}
	if refined == nil {
		t.Fatalf("LiveOptimizer returned nil patch")
	}
}
