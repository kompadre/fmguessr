package match

import "math"

// EliteCohort maintains a sorted, deduplicated list of the best candidates.
// Candidates are inserted incrementally as they are produced by each pipeline stage.
// Only candidates that beat the worst existing elite (and aren't duplicates) are added.
type EliteCohort struct {
	entries []EliteEntry
	maxLen  int
}

type EliteEntry struct {
	cand     *Candidate
	distance float64
}

// NewEliteCohort creates an empty cohort with the given maximum capacity.
func NewEliteCohort(maxLen int) *EliteCohort {
	return &EliteCohort{maxLen: maxLen}
}

// TryInsert attempts to add a candidate to the cohort.
// Returns true if the candidate was added (it improved the cohort).
//
// Insertion rules:
//   - Duplicate (by encoded patch signature) → skip
//   - Cohort not full → insert in sorted position
//   - Cohort full and new distance < worst distance → evict worst, insert new
//   - Otherwise → skip
func (ec *EliteCohort) TryInsert(c *Candidate, dist float64) bool {
	sig := string(c.Encode())

	// Check for duplicates
	for _, e := range ec.entries {
		if string(e.cand.Encode()) == sig {
			return false
		}
	}

	// If cohort is full and new candidate is worse than worst, skip
	if len(ec.entries) >= ec.maxLen && dist >= ec.entries[len(ec.entries)-1].distance {
		return false
	}

	// Find insertion position (sorted by distance ascending)
	pos := len(ec.entries)
	for i, e := range ec.entries {
		if dist < e.distance {
			pos = i
			break
		}
	}

	// Insert at position
	ec.entries = append(ec.entries, EliteEntry{})
	copy(ec.entries[pos+1:], ec.entries[pos:])
	ec.entries[pos] = EliteEntry{cand: c.Copy(), distance: dist}

	// Truncate to maxLen
	if len(ec.entries) > ec.maxLen {
		ec.entries = ec.entries[:ec.maxLen]
	}

	return true
}

// Candidates returns the elite candidates in ascending distance order.
func (ec *EliteCohort) Candidates() []*Candidate {
	result := make([]*Candidate, len(ec.entries))
	for i, e := range ec.entries {
		result[i] = e.cand
	}
	return result
}

// Worst returns the distance of the worst (highest distance) elite.
// Returns math.MaxFloat64 if the cohort is empty.
func (ec *EliteCohort) Worst() float64 {
	if len(ec.entries) == 0 {
		return math.MaxFloat64
	}
	return ec.entries[len(ec.entries)-1].distance
}

// Len returns the number of elites in the cohort.
func (ec *EliteCohort) Len() int {
	return len(ec.entries)
}
