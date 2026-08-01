package match_test

import (
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestOperatorMiniGA(t *testing.T) {
	// Create 4 distinct parent candidates
	p1 := match.SineBase()
	p2 := match.SquareBase()
	p3 := match.SawtoothBase()
	p4 := match.TriangleBase()

	parents := []*match.Candidate{p1, p2, p3, p4}

	// Render target from a hybrid candidate (Square carrier with Sawtooth modulator level)
	hybridTarget := match.SquareBase()
	hybridTarget.Ops[4].OutputLevel = 82 // Sawtooth modulator level
	targetAudio := match.RenderCandidate(hybridTarget, 60, 1*time.Second)
	targetNorm := match.NormalizeRMS(targetAudio)
	rawTargetRMS := match.CalcRMS(targetAudio)

	// Run operator mini-GA
	best := match.RunOperatorMiniGA(parents, targetNorm, 60, 1*time.Second, rawTargetRMS, 25, 32, true)

	if best == nil {
		t.Fatalf("RunOperatorMiniGA returned nil candidate")
	}

	t.Logf("Operator Mini-GA evolved hybrid candidate: BestDist=%.6f, Alg=%d", best.Fitness, best.Algorithm+1)

	if best.Fitness > 0.05 {
		t.Errorf("Operator Mini-GA failed to evolve close hybrid match: expected dist <= 0.05, got %.6f", best.Fitness)
	}
}
