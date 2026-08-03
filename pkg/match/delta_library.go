package match

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// DeltaEntry represents one meaningful step for a knob extracted from codebook presets.
type DeltaEntry struct {
	Delta     int  // Signed step size
	Frequency int  // How often this delta appears in codebook
	Rogue     bool // True if injected to fill a gap (not from codebook data)
}

// KnobDeltaLibrary holds all meaningful deltas for one knob in one role.
type KnobDeltaLibrary struct {
	Positions []int         // Unique sorted positions from codebook
	Median    int           // Median position (starting point for this knob)
	Range     [2]int        // [min, max] observed in codebook
	Deltas    []DeltaEntry  // Sorted by |Delta| ascending, then Frequency descending
}

// DeltaLibrary is the full extraction: role -> knob name -> deltas.
type DeltaLibrary struct {
	Carrier   map[string]*KnobDeltaLibrary
	Modulator map[string]*KnobDeltaLibrary
	rng       *rand.Rand
}

// BuildDeltaLibrary extracts parameter deltas from codebook entries, grouped by
// role (carrier/modulator) and knob name. Fills large gaps with rogue random deltas.
func BuildDeltaLibrary(cb *Codebook) *DeltaLibrary {
	lib := &DeltaLibrary{
		Carrier:   make(map[string]*KnobDeltaLibrary),
		Modulator: make(map[string]*KnobDeltaLibrary),
		rng:       rand.New(rand.NewSource(42)),
	}

	if cb == nil || len(cb.Entries) == 0 {
		return lib
	}

	// Collect per-(role, knob) position values from codebook
	type positionBucket struct {
		positions []int
	}
	buckets := make(map[string]*positionBucket) // key: "c:Op1.Level" or "m:Op3.Fine"

	for _, entry := range cb.Entries {
		algo := entry.Candidate.Algorithm
		for opIdx := 0; opIdx < 6; opIdx++ {
			role := "m"
			if IsCarrier(algo, opIdx) {
				role = "c"
			}
			opName := fmt.Sprintf("Op%d", opIdx+1)

			knobs := []struct {
				suffix string
				value  int
			}{
				{"Level", int(entry.Candidate.Ops[opIdx].OutputLevel)},
				{"Fine", int(entry.Candidate.Ops[opIdx].FreqFine)},
				{"Coarse", int(entry.Candidate.Ops[opIdx].FreqCoarse)},
			}

			for _, k := range knobs {
				key := role + ":" + opName + "." + k.suffix
				b, ok := buckets[key]
				if !ok {
					b = &positionBucket{}
					buckets[key] = b
				}
				b.positions = append(b.positions, k.value)
			}
		}

		// Feedback is role-independent
		fbKey := "c:Feedback"
		if _, ok := buckets[fbKey]; !ok {
			buckets[fbKey] = &positionBucket{}
		}
		buckets[fbKey].positions = append(buckets[fbKey].positions, int(entry.Candidate.Feedback))
		mfbKey := "m:Feedback"
		if _, ok := buckets[mfbKey]; !ok {
			buckets[mfbKey] = &positionBucket{}
		}
		buckets[mfbKey].positions = append(buckets[mfbKey].positions, int(entry.Candidate.Feedback))
	}

	// Build KnobDeltaLibrary for each bucket
	for key, bucket := range buckets {
		role := string(key[0])
		knobName := key[2:]

		libEntry := buildKnobDeltaLibrary(bucket.positions, lib.rng)

		if role == "c" {
			lib.Carrier[knobName] = libEntry
		} else {
			lib.Modulator[knobName] = libEntry
		}
	}

	return lib
}

// buildKnobDeltaLibrary computes deltas from a set of position values.
func buildKnobDeltaLibrary(positions []int, rng *rand.Rand) *KnobDeltaLibrary {
	if len(positions) == 0 {
		return &KnobDeltaLibrary{}
	}

	// Deduplicate and sort
	unique := make(map[int]bool)
	for _, p := range positions {
		unique[p] = true
	}
	sorted := make([]int, 0, len(unique))
	for p := range unique {
		sorted = append(sorted, p)
	}
	sort.Ints(sorted)

	median := sorted[len(sorted)/2]
	minVal := sorted[0]
	maxVal := sorted[len(sorted)-1]
	rangeSpan := maxVal - minVal

	// Compute all pairwise deltas
	deltaCounts := make(map[int]int)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			d := sorted[j] - sorted[i]
			deltaCounts[d]++
			deltaCounts[-d]++
		}
	}

	// Group deltas by magnitude into bins
	type deltaBin struct {
		deltas map[int]int // delta value -> count
	}
	bins := []deltaBin{
		{deltas: make(map[int]int)}, // 1
		{deltas: make(map[int]int)}, // 2-3
		{deltas: make(map[int]int)}, // 4-5
		{deltas: make(map[int]int)}, // 6-10
		{deltas: make(map[int]int)}, // 11-20
		{deltas: make(map[int]int)}, // 21+
	}

	for d, count := range deltaCounts {
		absD := int(math.Abs(float64(d)))
		switch {
		case absD <= 1:
			bins[0].deltas[d] = count
		case absD <= 3:
			bins[1].deltas[d] = count
		case absD <= 5:
			bins[2].deltas[d] = count
		case absD <= 10:
			bins[3].deltas[d] = count
		case absD <= 20:
			bins[4].deltas[d] = count
		default:
			bins[5].deltas[d] = count
		}
	}

	// For each bin, pick the most frequent signed delta
	var deltas []DeltaEntry
	for _, bin := range bins {
		if len(bin.deltas) == 0 {
			continue
		}
		type dEntry struct {
			delta int
			count int
		}
		var entries []dEntry
		for d, c := range bin.deltas {
			entries = append(entries, dEntry{d, c})
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].count > entries[j].count
		})
		// Take top 2 from each bin (positive and negative if both exist)
		taken := make(map[int]bool)
		for _, e := range entries {
			if taken[e.delta] {
				continue
			}
			deltas = append(deltas, DeltaEntry{Delta: e.delta, Frequency: e.count})
			taken[e.delta] = true
			// Also take the opposite sign if present
			opposite := -e.delta
			if !taken[opposite] {
				if _, ok := bin.deltas[opposite]; ok {
					deltas = append(deltas, DeltaEntry{Delta: opposite, Frequency: bin.deltas[opposite]})
					taken[opposite] = true
				}
			}
			if len(taken) >= 4 {
				break
			}
		}
	}

	// Rogue injection: fill gaps > 10% of range where no delta covers
	if rangeSpan > 5 {
		rogueThreshold := float64(rangeSpan) * 0.10
		sortedDeltas := make([]int, 0, len(deltaCounts))
		for d := range deltaCounts {
			sortedDeltas = append(sortedDeltas, int(math.Abs(float64(d))))
		}
		sort.Ints(sortedDeltas)

		// Find covered ranges
		covered := make([]bool, rangeSpan+1)
		for _, d := range sortedDeltas {
			if d <= rangeSpan {
				covered[d] = true
			}
		}

		// Find gaps in coverage
		var gaps [][2]int
		gapStart := -1
		for d := 1; d <= rangeSpan; d++ {
			if !covered[d] {
				if gapStart == -1 {
					gapStart = d
				}
			} else {
				if gapStart != -1 {
					gaps = append(gaps, [2]int{gapStart, d - 1})
					gapStart = -1
				}
			}
		}
		if gapStart != -1 {
			gaps = append(gaps, [2]int{gapStart, rangeSpan})
		}

		// Inject rogues for large gaps
		for _, gap := range gaps {
			gapSize := gap[1] - gap[0] + 1
			if float64(gapSize) >= rogueThreshold {
				// Pick a representative delta in the middle of the gap
				rogueDelta := (gap[0] + gap[1]) / 2
				deltas = append(deltas, DeltaEntry{
					Delta:     rogueDelta,
					Frequency: 1,
					Rogue:     true,
				})
				deltas = append(deltas, DeltaEntry{
					Delta:     -rogueDelta,
					Frequency: 1,
					Rogue:     true,
				})
			}
		}
	}

	// Sort by |Delta| ascending, then Frequency descending
	sort.Slice(deltas, func(i, j int) bool {
		absI := int(math.Abs(float64(deltas[i].Delta)))
		absJ := int(math.Abs(float64(deltas[j].Delta)))
		if absI != absJ {
			return absI < absJ
		}
		return deltas[i].Frequency > deltas[j].Frequency
	})

	return &KnobDeltaLibrary{
		Positions: sorted,
		Median:    median,
		Range:     [2]int{minVal, maxVal},
		Deltas:    deltas,
	}
}

// GetDeltas returns the delta list for a given role and knob name.
// Falls back to default deltas if no codebook data exists.
func (lib *DeltaLibrary) GetDeltas(role string, knobName string) []DeltaEntry {
	if lib == nil {
		return defaultDeltas()
	}

	var entry *KnobDeltaLibrary
	if role == "carrier" {
		entry = lib.Carrier[knobName]
	} else {
		entry = lib.Modulator[knobName]
	}

	if entry == nil || len(entry.Deltas) == 0 {
		return defaultDeltas()
	}
	return entry.Deltas
}

// GetMedian returns the codebook median for a knob, or a sensible default.
func (lib *DeltaLibrary) GetMedian(role string, knobName string) int {
	if lib == nil {
		return 0
	}

	var entry *KnobDeltaLibrary
	if role == "carrier" {
		entry = lib.Carrier[knobName]
	} else {
		entry = lib.Modulator[knobName]
	}

	if entry == nil {
		return 0
	}
	return entry.Median
}

// defaultDeltas returns fallback deltas when no codebook is available.
func defaultDeltas() []DeltaEntry {
	return []DeltaEntry{
		{Delta: 1, Frequency: 1},
		{Delta: -1, Frequency: 1},
		{Delta: 2, Frequency: 1},
		{Delta: -2, Frequency: 1},
		{Delta: 5, Frequency: 1},
		{Delta: -5, Frequency: 1},
		{Delta: 10, Frequency: 1},
		{Delta: -10, Frequency: 1},
	}
}
