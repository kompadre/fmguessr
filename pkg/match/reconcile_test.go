package match

import (
	"math"
	"testing"
)

func TestReconcileBest(t *testing.T) {
	var rc ResultsCohort
	if rc.Best != EmptyCandidate {
		t.Fatal("initial Best should be EmptyCandidate")
	}
	other := ResultsCohort{}
	other.Best = *SineBase()
	other.Best.Fitness = 0.5
	rc.Reconcile(other)
	if rc.Best.Fitness != 0.5 {
		t.Fatalf("expected Best.Fitness=0.5, got %f", rc.Best.Fitness)
	}
	other2 := ResultsCohort{}
	other2.Best = *SineBase()
	other2.Best.Fitness = 0.8
	rc.Reconcile(other2)
	if rc.Best.Fitness != 0.5 {
		t.Fatalf("expected Best.Fitness=0.5, got %f", rc.Best.Fitness)
	}
	other3 := ResultsCohort{}
	other3.Best = *SineBase()
	other3.Best.Fitness = 0.3
	rc.Reconcile(other3)
	if rc.Best.Fitness != 0.3 {
		t.Fatalf("expected Best.Fitness=0.3, got %f", rc.Best.Fitness)
	}
}

func TestReconcileTimbral(t *testing.T) {
	var rc ResultsCohort
	other := ResultsCohort{}
	for i := 0; i < 3; i++ {
		c := *SineBase()
		c.Ops[5].OutputLevel = uint8(99 - i*10)
		c.Fitness = float64(i) * 0.1
		c.Scores[ScoreTimbre] = float64(i) * 0.1
		other.Timbral[i] = c
	}
	other.TimbralN = 3
	rc.Reconcile(other)
	if rc.TimbralN != 3 {
		t.Fatalf("expected TimbralN=3, got %d", rc.TimbralN)
	}
	for i := 0; i < rc.TimbralN-1; i++ {
		if rc.Timbral[i].Scores[ScoreTimbre] > rc.Timbral[i+1].Scores[ScoreTimbre] {
			t.Fatalf("Timbral not sorted")
		}
	}
}

func TestReconcileDedup(t *testing.T) {
	var rc ResultsCohort
	c1 := *SineBase()
	c1.Fitness = 0.5
	c1.Scores[ScoreTimbre] = 0.5
	other := ResultsCohort{}
	other.Timbral[0] = c1
	other.TimbralN = 1
	rc.Reconcile(other)
	other2 := ResultsCohort{}
	other2.Timbral[0] = c1
	other2.TimbralN = 1
	rc.Reconcile(other2)
	if rc.TimbralN != 1 {
		t.Fatalf("expected TimbralN=1 (dedup), got %d", rc.TimbralN)
	}
}

func TestReconcileCap(t *testing.T) {
	var rc ResultsCohort
	for j := 0; j < MaxCohortLen+2; j++ {
		other := ResultsCohort{}
		c := *SineBase()
		c.Ops[5].OutputLevel = uint8(j % 100)
		c.Fitness = float64(j) * 0.01
		c.Scores[ScoreTimbre] = float64(j) * 0.01
		other.Timbral[0] = c
		other.TimbralN = 1
		rc.Reconcile(other)
	}
	if rc.TimbralN != MaxCohortLen {
		t.Fatalf("expected TimbralN=%d, got %d", MaxCohortLen, rc.TimbralN)
	}
	for i := 0; i < rc.TimbralN; i++ {
		if math.IsNaN(rc.Timbral[i].Scores[ScoreTimbre]) {
			t.Fatalf("Timbral[%d] has NaN score", i)
		}
	}
}

func TestReconcileEmptyCandidateIgnored(t *testing.T) {
	var rc ResultsCohort
	other := ResultsCohort{}
	other.Timbral[0] = EmptyCandidate
	other.TimbralN = 1
	rc.Reconcile(other)
	if rc.TimbralN != 0 {
		t.Fatalf("expected TimbralN=0, got %d", rc.TimbralN)
	}
}
