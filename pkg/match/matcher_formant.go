package match

import (
	"context"
	"fmt"
)

type FormantGenerate struct{}

func (s *FormantGenerate) Name() string { return "FormantGenerate" }

func (s *FormantGenerate) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	cfg := in.Config
	target := in.Target
	features := in.Features

	targetFormants := ExtractFormants(target, RenderSampleRate, 4)
	if len(targetFormants) == 0 || features.F0 <= 20 {
		return result
	}
	in.State.FormantPeaks = targetFormants

	formantNoteFreq := features.F0
	if cfg.NoteFreq > 0 {
		formantNoteFreq = cfg.NoteFreq
	}
	formantCandidates := GenerateFormantCandidates(targetFormants, formantNoteFreq, cfg.MIDINote)
	if len(formantCandidates) == 0 {
		return result
	}

	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}

	for _, fc := range formantCandidates {
		fcFits, _ := evaluateBatchParallel(ctx, []*Candidate{fc}, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
		fc.Fitness = fcFits[0]
		fc.Scores[ScoreFormant] = fcFits[0]
	}

	for i := 0; i < len(formantCandidates) && i < MaxCohortLen; i++ {
		result.Cohort.Formant[i] = *formantCandidates[i]
	}
	if len(formantCandidates) < MaxCohortLen {
		result.Cohort.FormantN = len(formantCandidates)
	} else {
		result.Cohort.FormantN = MaxCohortLen
	}

	bestFormant := formantCandidates[0]
	for _, fc := range formantCandidates {
		if fc.Fitness < bestFormant.Fitness {
			bestFormant = fc
		}
	}
	result.Cohort.Best = *bestFormant

	if cfg.Verbose {
		fmt.Printf("Formant generation: %d candidates from %d formants\n",
			len(formantCandidates), len(targetFormants))
	}
	return result
}
