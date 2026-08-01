package match_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestPureWaveformTemplates(t *testing.T) {
	seeds := match.DefaultWaveformSeeds()
	if len(seeds) != 5 {
		t.Fatalf("Expected 5 default waveform seeds, got %d", len(seeds))
	}

	sine := match.SineBase()
	square := match.SquareBase()
	saw := match.SawtoothBase()
	tri := match.TriangleBase()

	if sine == nil || square == nil || saw == nil || tri == nil {
		t.Fatalf("Failed to instantiate pure waveform shape templates")
	}

	// Render each pure waveform
	dur := 500 * time.Millisecond
	sineAudio := match.RenderCandidate(sine, 60, dur)
	squareAudio := match.RenderCandidate(square, 60, dur)
	sawAudio := match.RenderCandidate(saw, 60, dur)
	triAudio := match.RenderCandidate(tri, 60, dur)

	if len(sineAudio) == 0 || len(squareAudio) == 0 || len(sawAudio) == 0 || len(triAudio) == 0 {
		t.Fatalf("Pure waveform render returned empty audio")
	}

	sineRMS := match.CalcRMS(sineAudio)
	squareRMS := match.CalcRMS(squareAudio)
	sawRMS := match.CalcRMS(sawAudio)
	triRMS := match.CalcRMS(triAudio)

	t.Logf("Pure Waveform RMS: Sine=%.4f, Square=%.4f, Saw=%.4f, Tri=%.4f",
		sineRMS, squareRMS, sawRMS, triRMS)

	if sineRMS < 0.05 || squareRMS < 0.05 || sawRMS < 0.05 || triRMS < 0.05 {
		t.Errorf("Pure waveform template produced near-zero audio output")
	}
}

func TestWaveformSeedsOptimization(t *testing.T) {
	rng := rand.New(rand.NewSource(999))
	_ = rng

	// Generate target audio from SquareBase
	targetSquare := match.RenderCandidate(match.SquareBase(), 60, 1*time.Second)

	// Run LiveOptimizer starting from neutral SineBase
	liveOpt := match.NewLiveOptimizer(targetSquare, match.SineBase(), 10, true, 60, 1*time.Second)
	best := liveOpt.Run()

	t.Logf("LiveOptimizer starting from Sine on Square target: BestDist=%.6f", liveOpt.BestDist())

	if liveOpt.BestDist() > 0.05 {
		t.Errorf("LiveOptimizer failed to converge to Square base, dist: %.6f", liveOpt.BestDist())
	}
	if best == nil {
		t.Fatalf("LiveOptimizer returned nil patch")
	}
}
