package match

import (
	"context"
	"math"
)

type FormantRefineStep struct{}

func (s *FormantRefineStep) Name() string { return "FormantRefineStep" }

func (s *FormantRefineStep) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target

	rawTargetRMS := in.Features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	targetFormants := in.State.FormantPeaks
	if len(targetFormants) == 0 {
		targetFormants = ExtractFormants(target, RenderSampleRate, 4)
	}
	if len(targetFormants) == 0 {
		return result
	}

	noteFreq := 440.0 * math.Pow(2, float64(cfg.MIDINote-69)/12.0)

	var refineSources []*Candidate
	if in.Cohort.Best != EmptyCandidate {
		refineSources = append(refineSources, &in.Cohort.Best)
	}
	for _, nn := range in.State.NNCandidates {
		refineSources = append(refineSources, nn)
	}

	type formantResult struct {
		cand        *Candidate
		formantDist float64
		melDist     float64
	}
	var results []formantResult
	seenFormant := make(map[string]bool)

	for _, src := range refineSources {
		if ctx.Err() != nil {
			break
		}
		sig := string(src.Encode())
		if seenFormant[sig] {
			continue
		}
		seenFormant[sig] = true

		refined, fDist := FormantRefine(
			src, targetFormants,
			noteFreq, cfg.MIDINote, cfg.RenderDur,
			target, rawTargetRMS, in.TargetMel,
			src.Fitness, 0.10, 5, false,
		)
		refAudio := RenderCandidate(refined, cfg.MIDINote, cfg.RenderDur)
		refNorm := NormalizeRMS(TrimOrPad(refAudio, len(target)))
		mDist := MelCosineDistanceCached(in.TargetMel, refNorm, RenderSampleRate)

		if mDist <= src.Fitness*1.10 {
			results = append(results, formantResult{refined, fDist, mDist})
		}
	}

	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].formantDist < results[i].formantDist {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	for i := 0; i < len(results) && i < MaxCohortLen; i++ {
		result.Cohort.Formant[i] = *results[i].cand
		result.Cohort.Formant[i].Scores[ScoreFormant] = results[i].formantDist
		result.Cohort.Formant[i].Fitness = results[i].melDist
	}
	if len(results) < MaxCohortLen {
		result.Cohort.FormantN = len(results)
	} else {
		result.Cohort.FormantN = MaxCohortLen
	}

	if len(results) > 0 {
		formantBestDist := evaluateSingleCandidate(results[0].cand, target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur, in.TargetMel)
		if in.Cohort.Best == EmptyCandidate || formantBestDist < in.Cohort.Best.Fitness {
			result.Cohort.Best = *results[0].cand
			result.Cohort.Best.Fitness = formantBestDist
		}
	}
	return result
}
