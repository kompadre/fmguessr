package match

import (
	"context"
	"fmt"
	"sort"
)

type NNPredict struct{}

func (s *NNPredict) Name() string { return "NNPredict" }

func (s *NNPredict) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	if len(cfg.ModelData) == 0 && cfg.ModelPath == "" {
		return result
	}
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	timbreSpec := ExtractTimbreWindow(target, RenderSampleRate)

	var probePredictor *ProbePredictor
	var err error
	if len(cfg.ModelData) > 0 {
		probePredictor, err = NewProbePredictorFromBytes(cfg.ModelData)
	} else {
		probePredictor, err = NewProbePredictor(cfg.ModelPath)
	}
	if err != nil {
		fmt.Printf("Warning: could not load NN model: %v\n", err)
		result.Err = err
		return result
	}
	defer probePredictor.Close()

	predicted, probeResult := probePredictor.PredictToDirectCandidate(timbreSpec, int(cfg.MIDINote))
	if in.TimbreCB != nil {
		predicted = AttachNearestRawBaseTimbre(predicted, in.TimbreCB)
	} else if in.Codebook != nil {
		predicted = AttachNearestRawBase(predicted, in.Codebook)
	}
	if cfg.Verbose {
		fmt.Printf("NN probe prediction: algo=%d feedback=%d brightness=%.2f harmonicity=%.2f\n",
			predicted.Algorithm, predicted.Feedback, probeResult.Brightness, probeResult.Harmonicity)
	}

	type algoScore struct {
		algo uint8
		dist float64
		cand *Candidate
	}
	var scores []algoScore
	for algo := uint8(0); algo < 32; algo++ {
		cand := predicted.Copy()
		cand.Algorithm = algo
		if in.TimbreCB != nil {
			cand = AttachNearestRawBaseTimbre(cand, in.TimbreCB)
		} else if in.Codebook != nil {
			cand = AttachNearestRawBase(cand, in.Codebook)
		}
		fits, _ := evaluateBatchParallel([]*Candidate{cand}, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
		cand.Fitness = fits[0]
		scores = append(scores, algoScore{algo, fits[0], cand})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].dist < scores[j].dist })

	nnCandidates := make([]*Candidate, 0, 4)
	for i := 0; i < 4 && i < len(scores); i++ {
		nnCandidates = append(nnCandidates, scores[i].cand.Copy())
	}
	in.State.NNCandidates = nnCandidates

	if len(scores) > 0 {
		result.Cohort.Best = *scores[0].cand
		result.Cohort.Best.Fitness = scores[0].dist
		result.Cohort.Timbral[0] = *scores[0].cand
		result.Cohort.Timbral[0].Fitness = scores[0].dist
		result.Cohort.TimbralN = 1
	}
	if scores[0].dist <= 1e-4 {
		logElapsed(cfg, "Exact NN prediction found (dist=%.6f). Returning early!\n", scores[0].dist)
		result.EarlyReturn = true
		return result
	}
	return result
}
