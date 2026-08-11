package match

import (
	"context"
	"fmt"
)

type BeamSearch struct{}

func (s *BeamSearch) Name() string { return "BeamSearch" }

func (s *BeamSearch) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	if !cfg.UseBeamSearch {
		return result
	}
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	var opCache *OperatorCache
	var beamCB *Codebook
	if in.Codebook != nil && len(in.Codebook.Entries) > 0 {
		beamCB = in.Codebook
		opCache = BuildOperatorCache(in.Codebook)
	}
	if opCache == nil {
		opCache = BuildOperatorCacheFromWaveformSeeds()
	}

	targetPeakShape := ExtractPeakShapeFromAudio(target, RenderSampleRate)
	targetCachedSTFT := ComputeTargetSTFT(target)

	beamWidth := cfg.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 4
	}
	beamRenders := cfg.BeamMaxRenders
	if beamRenders <= 0 {
		beamRenders = 20000
	}

	beamResult := RunConstructiveBeam(
		ctx,
		target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur,
		opCache, beamCB, in.TimbreCB, targetPeakShape, targetCachedSTFT,
		beamWidth, beamRenders,
		func(s string) { if cfg.Verbose { fmt.Println(s) } },
	)

	if beamResult != nil {
		beamResult.Final.Fitness = beamResult.Trace[len(beamResult.Trace)-1].Distance
		result.Cohort.Best = beamResult.Final
		result.Cohort.Timbral[0] = beamResult.Final
		result.Cohort.TimbralN = 1
		in.State.BeamResult = beamResult
	}
	return result
}
