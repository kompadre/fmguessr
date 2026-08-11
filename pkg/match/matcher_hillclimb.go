package match

import (
	"context"
	"fmt"
	"math"
)

type HillClimbStep struct{}

func (s *HillClimbStep) Name() string { return "HillClimbStep" }

func (s *HillClimbStep) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	if in.Cohort.Best == EmptyCandidate || in.Cohort.Best.Fitness >= math.MaxFloat64 {
		return result
	}

	best := in.Cohort.Best
	gaBestDist := best.Fitness

	liveOpt := NewLiveOptimizer(ctx, target, &best, 500, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS).WithTargetMel(in.TargetMel)
	finalPatch := liveOpt.Run()
	finalDist := liveOpt.bestDist

	diffOpt := NewDiffOptimizer(ctx, target, finalPatch, 5, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	diffPatch := diffOpt.RunAlternating(3)
	if diffOpt.bestDist < finalDist {
		finalPatch = diffPatch
		finalDist = diffOpt.bestDist
	}

	if cfg.Verbose {
		fmt.Printf("[%dms] Phase 3 final distance: %.6f (GA was: %.6f)\n", logMs(), finalDist, gaBestDist)
	}

	finalDistBlended := evaluateSingleCandidate(finalPatch, target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur, in.TargetMel)
	if finalDistBlended < gaBestDist {
		result.Cohort.Best = *finalPatch
		result.Cohort.Best.Fitness = finalDistBlended
	}
	return result
}
