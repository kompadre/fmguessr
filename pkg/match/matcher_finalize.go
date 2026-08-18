package match

import (
	"fmt"
	"math"
	"sort"
)

func finalize(
	cohort *ResultsCohort,
	pool *CandidatePool,
	state *MatcherState,
	target []float64,
	features TargetFeatures,
	cachedTargetMel *PrecomputedTargetMel,
	cfg Config,
) Result {
	rawTargetRMS := features.TargetRMS
	if rawTargetRMS <= 0 {
		rawTargetRMS = CalcRMS(target)
	}
	best := cohort.Best
	if best == EmptyCandidate || best.Fitness >= math.MaxFloat64 {
		poolBest := pool.Best()
		if poolBest != EmptyCandidate {
			best = poolBest
		}
	}
	if best == EmptyCandidate || best.Fitness >= math.MaxFloat64 {
		best = *DecodeVoicePatch(epiano)
		best.Fitness = evaluateSingleCandidate(&best, target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur, cachedTargetMel)
	}

	seen := make(map[string]bool)
	var topN []*Candidate
	addUnique := func(c *Candidate) {
		if c == nil {
			return
		}
		if c.Fitness >= math.MaxFloat64 {
			piano := DecodeVoicePatch(epiano)
			piano.Fitness = evaluateSingleCandidate(piano, target, rawTargetRMS, cfg.MIDINote, cfg.RenderDur, cachedTargetMel)
			c = piano
		}
		sig := string(c.Encode())
		if !seen[sig] {
			seen[sig] = true
			topN = append(topN, c)
		}
	}

	addUnique(&best)
	for i := 0; i < cohort.TimbralN; i++ {
		addUnique(&cohort.Timbral[i])
	}
	for i := 0; i < cohort.FormantN; i++ {
		addUnique(&cohort.Formant[i])
	}
	for _, nc := range state.NNCandidates {
		addUnique(nc)
	}
	for i := 0; i < pool.N; i++ {
		c := pool.Slots[i]
		if c != EmptyCandidate {
			addUnique(&c)
		}
	}

	// Sort by Fitness (mel distance) so top-32 is ordered by quality
	sort.Slice(topN, func(i, j int) bool {
		return topN[i].Fitness < topN[j].Fitness
	})

	if len(topN) > 32 {
		topN = topN[:32]
	}

	best.CalibrateCarrierLevels(rawTargetRMS, cfg.MIDINote, cfg.RenderDur)
	for _, c := range topN {
		c.CalibrateCarrierLevels(rawTargetRMS, cfg.MIDINote, cfg.RenderDur)
	}

	if cfg.Ctx == nil || cfg.Ctx.Err() == nil {
		targetSpec := ExtractTemporalTimbreSpec(target, RenderSampleRate)
		for i := 0; i < cohort.TimbralN; i++ {
			c := &cohort.Timbral[i]
			if !math.IsNaN(c.Scores[ScoreTemporal]) {
				continue
			}
			pcm := RenderCandidate(c, cfg.MIDINote, cfg.RenderDur)
			norm := NormalizeRMS(pcm)
			spec := ExtractTemporalTimbreSpec(norm, RenderSampleRate)
			dist := TemporalTimbreDistance(targetSpec, spec)
			c.Scores[ScoreTemporal] = dist
			insertIntoCohort(cohort.Full[:], &cohort.FullN, *c, ScoreTemporal)
		}
	}

	timbralTop5 := make([]*Candidate, 0, cohort.TimbralN)
	for i := 0; i < cohort.TimbralN; i++ {
		c := cohort.Timbral[i]
		timbralTop5 = append(timbralTop5, &c)
	}
	fullTop5 := make([]*Candidate, 0, cohort.FullN)
	for i := 0; i < cohort.FullN; i++ {
		c := cohort.Full[i]
		fullTop5 = append(fullTop5, &c)
	}
	formantTop5 := make([]*Candidate, 0, cohort.FormantN)
	for i := 0; i < cohort.FormantN; i++ {
		c := cohort.Formant[i]
		formantTop5 = append(formantTop5, &c)
	}

	favorites := findFavorites(timbralTop5, fullTop5)
	if cfg.Verbose && len(favorites) > 0 {
		fmt.Printf("Favorites (in both cohorts): %d candidates\n", len(favorites))
	}

	reportProgress(cfg, 100, "Completed")

	return Result{
		Best:        &best,
		BestDist:    best.Fitness,
		Iters:       pool.N,
		Top:         topN,
		TimbralTop5: timbralTop5,
		FullTop5:    fullTop5,
		Favorites:   favorites,
		FormantTop5: formantTop5,
		SeenBefore:  state.SeenBefore,
		BestMatch:   state.BestMatch,
	}
}
