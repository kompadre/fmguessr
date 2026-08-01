package match_test

import (
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestCalcDBPenalty(t *testing.T) {
	targetRMS := 0.10 // -20 dB

	// Candidate with matching RMS -> 0 penalty
	p0 := match.CalcDBPenalty(0.10, targetRMS)
	if p0 != 0.0 {
		t.Errorf("Expected 0 penalty for matching RMS, got %.4f", p0)
	}

	// Candidate within 6 dB (e.g. RMS 0.06 = -24.4 dB, drop = 4.4 dB) -> 0 penalty
	p6 := match.CalcDBPenalty(0.06, targetRMS)
	if p6 != 0.0 {
		t.Errorf("Expected 0 penalty for <= 6 dB drop, got %.4f", p6)
	}

	// Candidate with 12 dB drop (RMS 0.025 = -32 dB) -> penalty > 0
	p12 := match.CalcDBPenalty(0.025, targetRMS)
	if p12 <= 0.5 {
		t.Errorf("Expected penalty > 0.5 for 12 dB drop, got %.4f", p12)
	}

	// Candidate near silent (RMS 1e-5) -> heavy penalty (10.0)
	pSilent := match.CalcDBPenalty(1e-5, targetRMS)
	if pSilent < 9.0 {
		t.Errorf("Expected hard penalty >= 9.0 for silent candidate, got %.4f", pSilent)
	}
}

func TestCalibrateCarrierLevels(t *testing.T) {
	// Generate a candidate with low carrier output level
	cand := match.HeuristicCandidateWithAlgo(match.TargetFeatures{
		F0:           440,
		F0Confidence: 1.0,
		SustainLevel: 0.8,
	}, 440, 31)

	// Artificially lower carrier output level
	for op := 0; op < 6; op++ {
		if match.IsCarrier(31, op) {
			cand.Ops[op].OutputLevel = 60
		}
	}

	targetRMS := 0.12 // Target RMS
	dur := 1 * time.Second

	// Measure RMS before calibration
	audioBefore := match.RenderCandidate(cand, 69, dur)
	rmsBefore := match.CalcRMS(audioBefore)

	// Calibrate carrier levels
	cand.CalibrateCarrierLevels(targetRMS, 69, dur)

	// Measure RMS after calibration
	audioAfter := match.RenderCandidate(cand, 69, dur)
	rmsAfter := match.CalcRMS(audioAfter)

	t.Logf("RMS before calibration: %.6f, RMS after calibration: %.6f (target: %.6f)",
		rmsBefore, rmsAfter, targetRMS)

	if rmsAfter <= rmsBefore {
		t.Errorf("CalibrateCarrierLevels failed to boost quiet candidate: before=%.6f, after=%.6f",
			rmsBefore, rmsAfter)
	}
}
