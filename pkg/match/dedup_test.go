package match_test

import (
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestDeduplication(t *testing.T) {
	// 1. Test GA TopN Deduplication
	rng := rand.New(rand.NewSource(123))
	candA := match.RandomCandidate(rng)
	candB := match.RandomCandidate(rng)

	// Create a population with clones
	pop := []*match.Candidate{
		candA.Copy(),
		candA.Copy(),
		candB.Copy(),
		candA.Copy(),
		candB.Copy(),
	}
	fits := []float64{0.1, 0.1, 0.2, 0.1, 0.2}

	gaCfg := match.DefaultGAConfig()
	gaCfg.Population = len(pop)
	ga := match.NewGA(pop, fits, nil, gaCfg)

	top := ga.TopN(5)
	if len(top) != 2 {
		t.Errorf("Expected TopN to return 2 unique candidates, got %d", len(top))
	}

	// 2. Test Codebook Pure Patch Binary Deduplication across different entry names
	tempDir := t.TempDir()
	cbPath := filepath.Join(tempDir, "dedup_codebook.bin")

	// Index candA under name "Job_1"
	err := match.IndexCandidateToCodebook(candA, "Job_1", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("Failed to index Job_1: %v", err)
	}

	// Index identical candA under name "Job_2"
	err = match.IndexCandidateToCodebook(candA, "Job_2", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("Failed to index Job_2: %v", err)
	}

	cb, err := match.LoadCodebook(cbPath)
	if err != nil {
		t.Fatalf("Failed to load codebook: %v", err)
	}

	if len(cb.Entries) != 1 {
		t.Errorf("Codebook failed to deduplicate identical patch across different names: got %d entries, expected 1", len(cb.Entries))
	}
}
