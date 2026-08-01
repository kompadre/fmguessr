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
