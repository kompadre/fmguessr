package match

import "sort"

// CachedOpConfig is a pre-baked operator parameter set extracted from the codebook.
type CachedOpConfig struct {
	FreqCoarse  uint8
	FreqFine    uint8
	OutputLevel uint8
	Rate1       uint8
	Rate2       uint8
	Level3      uint8
	Rate4       uint8
}

// OperatorCache holds pre-baked operator configurations grouped by role and
// frequency ratio. Used by the beam search to seed operator evaluations with
// historically common parameter sets instead of random initialization.
type OperatorCache struct {
	Carriers   map[uint8][]CachedOpConfig // FreqCoarse -> configs (sorted by prevalence)
	Modulators map[uint8][]CachedOpConfig
	FeedbackPreset uint8
}

// BuildOperatorCache scans a codebook and extracts the most common operator
// parameter configurations, grouped by role (carrier/modulator) and FreqCoarse.
func BuildOperatorCache(cb *Codebook) *OperatorCache {
	cache := &OperatorCache{
		Carriers:   make(map[uint8][]CachedOpConfig),
		Modulators: make(map[uint8][]CachedOpConfig),
	}

	if cb == nil || len(cb.Entries) == 0 {
		return cache
	}

	// Collect param sets per (role, FreqCoarse) bucket
	type bucket struct {
		configs []CachedOpConfig
		counts  []int
	}
	buckets := make(map[string]*bucket) // key: "c:3" or "m:1"

	var feedbacks []uint8

	for _, entry := range cb.Entries {
		algo := entry.Candidate.Algorithm
		feedbacks = append(feedbacks, entry.Candidate.Feedback)

		for opIdx := 0; opIdx < 6; opIdx++ {
			role := "m"
			if IsCarrier(algo, opIdx) {
				role = "c"
			}
			fc := entry.Candidate.Ops[opIdx].FreqCoarse
			key := role + ":" + string(rune('0'+fc))

			b, ok := buckets[key]
			if !ok {
				b = &bucket{}
				buckets[key] = b
			}

			cfg := CachedOpConfig{
				FreqCoarse:  fc,
				FreqFine:    entry.Candidate.Ops[opIdx].FreqFine,
				OutputLevel: entry.Candidate.Ops[opIdx].OutputLevel,
				Rate1:       entry.Candidate.Ops[opIdx].Rate1,
				Rate2:       entry.Candidate.Ops[opIdx].Rate2,
				Level3:      entry.Candidate.Ops[opIdx].Level3,
				Rate4:       entry.Candidate.Ops[opIdx].Rate4,
			}

			// Check if this exact config already exists in the bucket
			found := false
			for i, existing := range b.configs {
				if existing.FreqFine == cfg.FreqFine &&
					existing.OutputLevel == cfg.OutputLevel &&
					existing.Rate2 == cfg.Rate2 &&
					existing.Level3 == cfg.Level3 {
					b.counts[i]++
					found = true
					break
				}
			}
			if !found {
				b.configs = append(b.configs, cfg)
				b.counts = append(b.counts, 1)
			}
		}
	}

	// Sort each bucket by frequency (most common first), keep top 10
	for key, b := range buckets {
		role := key[0]
		// Parse FreqCoarse from key (skip "c:" or "m:" prefix)
		fc := uint8(0)
		if len(key) > 2 {
			fc = uint8(key[2] - '0')
			// Handle FreqCoarse >= 10 (multi-digit)
			if len(key) > 3 && key[3] >= '0' && key[3] <= '9' {
				fc = fc*10 + uint8(key[3]-'0')
			}
		}

		// Sort by count descending
		indices := make([]int, len(b.counts))
		for i := range indices {
			indices[i] = i
		}
		sort.SliceStable(indices, func(i, j int) bool {
			return b.counts[indices[i]] > b.counts[indices[j]]
		})

		var sorted []CachedOpConfig
		limit := 10
		if len(indices) < limit {
			limit = len(indices)
		}
		for i := 0; i < limit; i++ {
			sorted = append(sorted, b.configs[indices[i]])
		}

		if role == 'c' {
			cache.Carriers[fc] = sorted
		} else {
			cache.Modulators[fc] = sorted
		}
	}

	// Compute median feedback
	if len(feedbacks) > 0 {
		sort.Slice(feedbacks, func(i, j int) bool { return feedbacks[i] < feedbacks[j] })
		cache.FeedbackPreset = feedbacks[len(feedbacks)/2]
	}

	return cache
}

// GetSeeds returns up to `count` parameter configurations for the given role
// and frequency ratio. Returns cached configs if available, otherwise returns
// sensible defaults.
func (cache *OperatorCache) GetSeeds(role string, freqCoarse uint8, count int) []OpParams {
	var configs []CachedOpConfig

	if cache != nil {
		if role == "carrier" {
			configs = cache.Carriers[freqCoarse]
		} else {
			configs = cache.Modulators[freqCoarse]
		}
	}

	var seeds []OpParams
	for i := 0; i < count && i < len(configs); i++ {
		c := configs[i]
		seeds = append(seeds, OpParams{
			Mode:        0, // ratio mode
			FreqCoarse:  c.FreqCoarse,
			FreqFine:    c.FreqFine,
			OutputLevel: c.OutputLevel,
			Rate1:       c.Rate1,
			Rate2:       c.Rate2,
			Level3:      c.Level3,
			Rate4:       c.Rate4,
		})
	}

	// Fill remaining seeds with defaults if cache didn't have enough
	defaultLevel := uint8(80)
	if role == "modulator" {
		defaultLevel = 65
	}
	for len(seeds) < count {
		seeds = append(seeds, OpParams{
			Mode:        0,
			FreqCoarse:  freqCoarse,
			FreqFine:    0,
			OutputLevel: defaultLevel,
			Rate1:       99,
			Rate2:       80,
			Level3:      90,
			Rate4:       40,
		})
	}

	return seeds
}

// GetFeedbackPreset returns the median feedback value from the codebook.
// Returns 0 if the cache is empty.
func (cache *OperatorCache) GetFeedbackPreset() uint8 {
	if cache == nil {
		return 0
	}
	return cache.FeedbackPreset
}

// BuildOperatorCacheFromWaveformSeeds builds a minimal cache from the 5 built-in
// waveform base templates when no codebook is available.
func BuildOperatorCacheFromWaveformSeeds() *OperatorCache {
	cache := &OperatorCache{
		Carriers:   make(map[uint8][]CachedOpConfig),
		Modulators: make(map[uint8][]CachedOpConfig),
	}

	bases := DefaultWaveformSeeds()
	for _, base := range bases {
		algo := base.Algorithm
		for opIdx := 0; opIdx < 6; opIdx++ {
			role := "modulator"
			if IsCarrier(algo, opIdx) {
				role = "carrier"
			}
			op := base.Ops[opIdx]
			if op.OutputLevel == 0 {
				continue // skip muted operators
			}
			cfg := CachedOpConfig{
				FreqCoarse:  op.FreqCoarse,
				FreqFine:    op.FreqFine,
				OutputLevel: op.OutputLevel,
				Rate1:       op.Rate1,
				Rate2:       op.Rate2,
				Level3:      op.Level3,
				Rate4:       op.Rate4,
			}

			bucket := &cache.Carriers
			if role == "modulator" {
				bucket = &cache.Modulators
			}
			fc := op.FreqCoarse
			existing := (*bucket)[fc]
			// Add if not duplicate
			dupe := false
			for _, e := range existing {
				if e.OutputLevel == cfg.OutputLevel && e.FreqFine == cfg.FreqFine {
					dupe = true
					break
				}
			}
			if !dupe {
				(*bucket)[fc] = append(existing, cfg)
			}
		}
	}

	cache.FeedbackPreset = 0
	return cache
}
