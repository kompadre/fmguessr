package match

import (
	"context"
	"fmt"
	"math"
)

type CodebookSearch struct{}

func (s *CodebookSearch) Name() string { return "CodebookSearch" }

func (s *CodebookSearch) Run(ctx context.Context, in MatcherInput) MatcherResult {
	result := MatcherResult{}
	if in.Config.CodebookPath == "" || in.Codebook == nil || len(in.Codebook.Entries) == 0 {
		return result
	}
	cb := in.Codebook
	target := in.Target
	cfg := in.Config
	rawTargetRMS := in.Features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	targetFp := ComputeFingerprint(target, RenderSampleRate)

	if in.TimbreCB != nil && len(in.TimbreCB.Entries) > 0 {
		timbreSpec := ExtractTimbreWindow(target, RenderSampleRate)
		const codebookTopK = 24
		fpMatches, _ := cb.SearchKNNWithDistances(targetFp, codebookTopK)
		timMatches, _ := in.TimbreCB.SearchKNNTimbreWithDistances(timbreSpec, codebookTopK)

		if len(fpMatches) > 0 {
			fpFits, _ := evaluateBatchParallel(ctx, fpMatches, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
			for i, c := range fpMatches {
				c.Scores[ScoreTimbre] = fpFits[i]
				c.Fitness = fpFits[i]
			}
			for i := 0; i < len(fpMatches) && i < MaxCohortLen; i++ {
				result.Cohort.Full[i] = *fpMatches[i]
			}
			if len(fpMatches) < MaxCohortLen {
				result.Cohort.FullN = len(fpMatches)
			} else {
				result.Cohort.FullN = MaxCohortLen
			}
			logElapsed(cfg, "Fingerprint codebook: evaluated %d candidates\n", len(fpMatches))
		}

		if len(timMatches) > 0 {
			timFits, _ := evaluateBatchParallel(ctx, timMatches, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
			for i, c := range timMatches {
				c.Scores[ScoreTimbre] = timFits[i]
				c.Fitness = timFits[i]
			}
			for i := 0; i < len(timMatches) && i < MaxCohortLen; i++ {
				result.Cohort.Timbral[i] = *timMatches[i]
			}
			if len(timMatches) < MaxCohortLen {
				result.Cohort.TimbralN = len(timMatches)
			} else {
				result.Cohort.TimbralN = MaxCohortLen
			}
			logElapsed(cfg, "Timbre codebook: evaluated %d candidates\n", len(timMatches))
		}

		bestCodebookDist := math.MaxFloat64
		var bestCodebookCand *Candidate
		allMatches := append(fpMatches, timMatches...)

		// Build knnMap: best candidate per algorithm for GA island seeding
		knnMap := make(map[uint8]*Candidate)
		knnMapDists := make(map[uint8]float64)
		for _, matchCand := range allMatches {
			dist := matchCand.Scores[ScoreTimbre]
			algo := matchCand.Algorithm
			if existing, ok := knnMapDists[algo]; !ok || dist < existing {
				knnMap[algo] = matchCand
				knnMapDists[algo] = dist
			}
			if dist < bestCodebookDist {
				bestCodebookDist = dist
				bestCodebookCand = matchCand.Copy()
				bestCodebookCand.Fitness = dist
			}
		}
		in.State.KNNMap = knnMap
		if bestCodebookCand != nil {
			result.Cohort.Best = *bestCodebookCand
		}

		// Populate SeenBefore + BestMatch from codebook results
		if bestCodebookDist <= 1e-3 {
			in.State.SeenBefore = true
		}
		if bestCodebookCand != nil {
			name := "CODEBOOK_PRESET"
			if in.TimbreCB != nil {
				for _, entry := range in.TimbreCB.Entries {
					if string(entry.Candidate.Encode()) == string(bestCodebookCand.Encode()) && entry.Name != "" {
						name = entry.Name
						break
					}
				}
			} else if in.Codebook != nil {
				for _, entry := range in.Codebook.Entries {
					if string(entry.Candidate.Encode()) == string(bestCodebookCand.Encode()) && entry.Name != "" {
						name = entry.Name
						break
					}
				}
			}
			in.State.BestMatch = &CodebookMatchDetails{
				Name:      name,
				Distance:  bestCodebookDist,
				Algorithm: int(bestCodebookCand.Algorithm + 1),
				Feedback:  int(bestCodebookCand.Feedback),
			}
		}
		if cfg.Verbose {
			fmt.Printf("Step 1 Dual-Codebook: FP=%d Timbre=%d → best dist=%.6f\n",
				len(fpMatches), len(timMatches), bestCodebookDist)
		}
		if bestCodebookDist <= 1e-4 {
			logElapsed(cfg, "Exact codebook match found (dist=%.6f). Returning early!\n", bestCodebookDist)
			result.EarlyReturn = true
			return result
		}
	} else {
		topK := 48
		if len(cb.Entries) < topK {
			topK = len(cb.Entries)
		}
		knnMatches := cb.SearchKNN(targetFp, topK)
		if len(knnMatches) > 0 {
			knnFits, _ := evaluateBatchParallel(ctx, knnMatches, target, cfg.MIDINote, cfg.RenderDur, rawTargetRMS, in.TargetMel)
			bestCodebookDist := math.MaxFloat64
			var bestCodebookCand *Candidate
			knnMap := make(map[uint8]*Candidate)
			knnMapDists := make(map[uint8]float64)
			for i, matchCand := range knnMatches {
				matchCand.Scores[ScoreTimbre] = knnFits[i]
				matchCand.Fitness = knnFits[i]
				algo := matchCand.Algorithm
				if existing, ok := knnMapDists[algo]; !ok || knnFits[i] < existing {
					knnMap[algo] = matchCand
					knnMapDists[algo] = knnFits[i]
				}
				if knnFits[i] < bestCodebookDist {
					bestCodebookDist = knnFits[i]
					bestCodebookCand = matchCand.Copy()
					bestCodebookCand.Fitness = knnFits[i]
				}
			}
			in.State.KNNMap = knnMap
			for i := 0; i < len(knnMatches) && i < MaxCohortLen; i++ {
				result.Cohort.Timbral[i] = *knnMatches[i]
			}
			if len(knnMatches) < MaxCohortLen {
				result.Cohort.TimbralN = len(knnMatches)
			} else {
				result.Cohort.TimbralN = MaxCohortLen
			}
			if bestCodebookCand != nil {
				result.Cohort.Best = *bestCodebookCand
			}

			// Populate SeenBefore + BestMatch from fingerprint codebook results
			if bestCodebookDist <= 1e-3 {
				in.State.SeenBefore = true
			}
			if bestCodebookCand != nil {
				name := "CODEBOOK_PRESET"
				for _, entry := range in.Codebook.Entries {
					if string(entry.Candidate.Encode()) == string(bestCodebookCand.Encode()) && entry.Name != "" {
						name = entry.Name
						break
					}
				}
				in.State.BestMatch = &CodebookMatchDetails{
					Name:      name,
					Distance:  bestCodebookDist,
					Algorithm: int(bestCodebookCand.Algorithm + 1),
					Feedback:  int(bestCodebookCand.Feedback),
				}
			}
			if cfg.Verbose {
				fmt.Printf("Step 1 Fingerprint-Only: evaluated %d candidates (best dist=%.6f)\n",
					len(knnMatches), bestCodebookDist)
			}
			if bestCodebookDist <= 1e-4 {
				logElapsed(cfg, "Exact codebook match found (dist=%.6f). Returning early!\n", bestCodebookDist)
				result.EarlyReturn = true
				return result
			}
		}
	}
	return result
}
