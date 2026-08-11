package match

import (
	"context"
	"fmt"
)

type CSPOAStep struct{}

func (s *CSPOAStep) Name() string { return "CSPOAStep" }

func (s *CSPOAStep) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	if !cfg.CSPOA.Enabled {
		return result
	}
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	cspoaCB := in.Codebook
	if cspoaCB == nil {
		return result
	}

	opCache := BuildOperatorCache(cspoaCB)
	var bestCodebookCand *Candidate
	if in.Cohort.Best != EmptyCandidate {
		bestCodebookCand = &in.Cohort.Best
	}

	deltaLib := BuildDeltaLibrary(cspoaCB)
	cspoaResults := RunCSPOA(
		ctx,
		target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur,
		deltaLib, opCache, bestCodebookCand, cfg.CSPOA,
		func(s string) { if cfg.Verbose { fmt.Println(s) } },
	)

	for i := 0; i < len(cspoaResults) && i < MaxCohortLen; i++ {
		result.Cohort.Timbral[i] = *cspoaResults[i]
	}
	if len(cspoaResults) < MaxCohortLen {
		result.Cohort.TimbralN = len(cspoaResults)
	} else {
		result.Cohort.TimbralN = MaxCohortLen
	}
	if len(cspoaResults) > 0 {
		result.Cohort.Best = *cspoaResults[0]
	}
	return result
}
