package match

import (
	"context"
	"fmt"
	"math"
)

type NNRefine struct{}

func (s *NNRefine) Name() string { return "NNRefine" }

func (s *NNRefine) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	if len(in.State.NNCandidates) == 0 {
		return result
	}
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	nnDist := in.State.NNCandidates[0].Fitness
	optThreshold := math.MaxFloat64
	if in.Cohort.Best != EmptyCandidate && in.Cohort.Best.Fitness < math.MaxFloat64 {
		optThreshold = in.Cohort.Best.Fitness * (1 + cfg.OptMargin)
	}
	if nnDist > optThreshold {
		if cfg.Verbose {
			fmt.Printf("Skipping NN pre-optimization: dist=%.6f > threshold=%.6f\n", nnDist, optThreshold)
		}
		return result
	}

	initialPatch := in.State.NNCandidates[0]
	if in.Cohort.Best != EmptyCandidate && in.Cohort.Best.Fitness < initialPatch.Fitness {
		initialPatch = &in.Cohort.Best
	}

	liveOpt := NewLiveOptimizer(ctx, target, initialPatch, 1000, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS).WithTargetMel(in.TargetMel)
	in.State.LiveOpt = liveOpt
	refined := liveOpt.Run()
	diffOpt := NewDiffOptimizer(ctx, target, refined, 5, cfg.Verbose, cfg.MIDINote, cfg.RenderDur, rawTargetRMS)
	diffRefined := diffOpt.RunAlternating(2)
	if diffOpt.bestDist < liveOpt.bestDist {
		refined = diffRefined
	}
	refinedDist := liveOpt.bestDist
	if diffOpt.bestDist < refinedDist {
		refinedDist = diffOpt.bestDist
	}

	in.State.LiveOptBest = refined
	result.Cohort.Best = *refined
	result.Cohort.Best.Fitness = refinedDist
	result.Cohort.Timbral[0] = *refined
	result.Cohort.Timbral[0].Fitness = refinedDist
	result.Cohort.TimbralN = 1

	if refinedDist < nnDist {
		in.State.NNCandidates[0] = refined
	}
	if refinedDist <= 1e-4 {
		logElapsed(cfg, "Exact match found during NN pre-optimization (dist=%.6f). Returning early!\n", refinedDist)
		result.EarlyReturn = true
		return result
	}
	return result
}
