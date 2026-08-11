package match

import (
	"context"
	"math"
)

const (
	MaxCohortLen = 5
	MaxPoolSize  = 1024
)

type ResultsCohort struct {
	Best     Candidate
	Timbral  [MaxCohortLen]Candidate
	Full     [MaxCohortLen]Candidate
	Formant  [MaxCohortLen]Candidate
	TimbralN int
	FullN    int
	FormantN int
}

type CandidatePool struct {
	Slots [MaxPoolSize]Candidate
	N     int
}

func (p *CandidatePool) Assign(c Candidate) int {
	if p.N >= MaxPoolSize {
		return -1
	}
	idx := p.N
	p.Slots[idx] = c
	p.N++
	return idx
}

func (p *CandidatePool) Best() Candidate {
	var best Candidate
	best.Fitness = math.MaxFloat64
	for i := 0; i < p.N; i++ {
		c := p.Slots[i]
		if c != EmptyCandidate && c.Fitness < best.Fitness {
			best = c
		}
	}
	if best.Fitness == math.MaxFloat64 {
		return EmptyCandidate
	}
	return best
}

type MatcherState struct {
	NNCandidates  []*Candidate
	FormantPeaks  []FormantPeak
	ClassifierMap map[uint8]int
	LiveOptBest   *Candidate
	LiveOpt       *LiveOptimizer // for permutation candidate generation in GA
	KNNMap        map[uint8]*Candidate // codebook best per algorithm, populated by CodebookSearch
	BeamResult    *PatchEvolution      // beam search result, populated by BeamSearch
}

type MatcherInput struct {
	Target    []float64
	Features  TargetFeatures
	Config    Config
	TargetMel *PrecomputedTargetMel
	Cohort    *ResultsCohort
	Pool      *CandidatePool
	Codebook  *Codebook
	TimbreCB  *TimbreCodebook
	State     *MatcherState
}

type MatcherResult struct {
	Cohort      ResultsCohort
	EarlyReturn bool
	Err         error
}

type Matcher interface {
	Name() string
	Run(ctx context.Context, in MatcherInput) MatcherResult
}

func (rc *ResultsCohort) Reconcile(other ResultsCohort) {
	if other.Best != EmptyCandidate && other.Best.Fitness < math.MaxFloat64 {
		if rc.Best == EmptyCandidate || other.Best.Fitness < rc.Best.Fitness {
			rc.Best = other.Best
		}
	}
	for i := 0; i < other.TimbralN; i++ {
		c := other.Timbral[i]
		if c == EmptyCandidate || c.Fitness >= math.MaxFloat64 {
			continue
		}
		insertIntoCohort(rc.Timbral[:], &rc.TimbralN, c, ScoreTimbre)
	}
	for i := 0; i < other.FullN; i++ {
		c := other.Full[i]
		if c == EmptyCandidate || c.Fitness >= math.MaxFloat64 {
			continue
		}
		insertIntoCohort(rc.Full[:], &rc.FullN, c, ScoreTemporal)
	}
	for i := 0; i < other.FormantN; i++ {
		c := other.Formant[i]
		if c == EmptyCandidate || c.Fitness >= math.MaxFloat64 {
			continue
		}
		insertIntoCohort(rc.Formant[:], &rc.FormantN, c, ScoreFormant)
	}
}

func insertIntoCohort(slice []Candidate, count *int, c Candidate, scoreIdx int) {
	dist := c.Scores[scoreIdx]
	if math.IsNaN(dist) {
		return
	}
	sig := string(c.Encode())
	for i := 0; i < *count; i++ {
		if string(slice[i].Encode()) == sig {
			return
		}
	}
	if *count >= MaxCohortLen && dist >= slice[*count-1].Scores[scoreIdx] {
		return
	}
	pos := *count
	for i := 0; i < *count; i++ {
		if dist < slice[i].Scores[scoreIdx] {
			pos = i
			break
		}
	}
	if *count < MaxCohortLen {
		*count++
	}
	for i := *count - 1; i > pos; i-- {
		slice[i] = slice[i-1]
	}
	slice[pos] = c
	if *count > MaxCohortLen {
		*count = MaxCohortLen
	}
}
