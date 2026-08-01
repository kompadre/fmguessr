package match_test

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestAutoIndexCandidateToCodebook(t *testing.T) {
	tempDir := t.TempDir()
	cbPath := filepath.Join(tempDir, "test_autoindex.bin")
	timbrePath := filepath.Join(tempDir, "test_autoindex_timbre.bin")

	rng := rand.New(rand.NewSource(999))
	cand := match.RandomCandidate(rng)

	// Index candidate
	err := match.IndexCandidateToCodebook(cand, "TEST_PATCH", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("IndexCandidateToCodebook failed: %v", err)
	}

	// Verify standard codebook file
	cb, err := match.LoadCodebook(cbPath)
	if err != nil {
		t.Fatalf("LoadCodebook failed: %v", err)
	}
	if len(cb.Entries) != 1 {
		t.Fatalf("Expected 1 codebook entry, got %d", len(cb.Entries))
	}
	if cb.Entries[0].Name != "TEST_PATCH" {
		t.Errorf("Expected entry name TEST_PATCH, got %s", cb.Entries[0].Name)
	}

	// Verify timbre codebook file
	tcb, err := match.LoadTimbreCodebook(timbrePath)
	if err != nil {
		t.Fatalf("LoadTimbreCodebook failed: %v", err)
	}
	if len(tcb.Entries) != 1 {
		t.Fatalf("Expected 1 timbre codebook entry, got %d", len(tcb.Entries))
	}
	if tcb.Entries[0].Name != "TEST_PATCH" {
		t.Errorf("Expected timbre entry name TEST_PATCH, got %s", tcb.Entries[0].Name)
	}

	// Verify deduplication: re-index identical candidate
	err = match.IndexCandidateToCodebook(cand, "TEST_PATCH", 60, 1*time.Second, 1*time.Second, cbPath)
	if err != nil {
		t.Fatalf("Re-indexing failed: %v", err)
	}

	cb, _ = match.LoadCodebook(cbPath)
	tcb, _ = match.LoadTimbreCodebook(timbrePath)
	if len(cb.Entries) != 1 {
		t.Errorf("Deduplication failed for standard codebook, count=%d", len(cb.Entries))
	}
	if len(tcb.Entries) != 1 {
		t.Errorf("Deduplication failed for timbre codebook, count=%d", len(tcb.Entries))
	}

	os.Remove(cbPath)
	os.Remove(timbrePath)
}
