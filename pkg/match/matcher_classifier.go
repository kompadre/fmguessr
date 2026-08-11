package match

import (
	"context"
	"math"
)

type ClassifierStep struct{}

func (s *ClassifierStep) Name() string { return "ClassifierStep" }

func (s *ClassifierStep) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target

	if cfg.ClassifierPath == "" {
		return result
	}
	rawTargetRMS := in.Features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	classifier, clsErr := NewPatchClassifier(cfg.ClassifierPath)
	if clsErr != nil {
		result.Err = clsErr
		return result
	}
	defer classifier.Close()

	melFeatures := ExtractMelSpectrogram(target, RenderSampleRate)
	pred := classifier.PredictTopK(melFeatures, 5)

	var refParams [DX7LabelDims]float64
	if len(pred.TopK) > 0 && pred.TopK[0].PatchBytes != nil {
		refCand := DecodeVoicePatch(pred.TopK[0].PatchBytes)
		if refCand != nil {
			refParams = candidateToParams53(refCand)
		}
	}

	classifierAlgoMap := make(map[uint8]int)
	var classifierCandidates []*Candidate
	bestClassifierDist := math.MaxFloat64
	var bestClassifierCand *Candidate

	for _, m := range pred.TopK {
		if m.PatchBytes == nil {
			continue
		}
		cand := DecodeVoicePatch(m.PatchBytes)
		if cand == nil {
			continue
		}
		candAudio := RenderCandidate(cand, cfg.MIDINote, cfg.RenderDur)
		candTrimmed := TrimOrPad(candAudio, len(target))
		candNormed := NormalizeRMS(candTrimmed)
		candParams := candidateToParams53(cand)
		combinedDist := CombinedDistance(target, candNormed, RenderSampleRate, refParams, candParams)
		fullDist := evaluateSingleCandidate(cand, target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur, in.TargetMel)
		cand.Fitness = fullDist
		classifierCandidates = append(classifierCandidates, cand)
		classifierAlgoMap[cand.Algorithm] = len(classifierCandidates) - 1
		if fullDist < bestClassifierDist {
			bestClassifierDist = fullDist
			bestClassifierCand = cand
		}
		_ = combinedDist
	}

	in.State.ClassifierMap = classifierAlgoMap
	in.State.NNCandidates = append(in.State.NNCandidates, classifierCandidates...)

	for i := 0; i < len(classifierCandidates) && i < MaxCohortLen; i++ {
		result.Cohort.Timbral[i] = *classifierCandidates[i]
	}
	if len(classifierCandidates) < MaxCohortLen {
		result.Cohort.TimbralN = len(classifierCandidates)
	} else {
		result.Cohort.TimbralN = MaxCohortLen
	}
	if bestClassifierCand != nil {
		result.Cohort.Best = *bestClassifierCand
	}

	if bestClassifierCand != nil && bestClassifierDist < 0.10 && pred.Confidence >= 0.99 {
		result.EarlyReturn = true
	}
	return result
}
