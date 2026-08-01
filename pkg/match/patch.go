package match

import (
	"fmt"
	"math"
	"math/rand"
)

// epiano is the default DX7 voice used as a base for encoding (128 bytes).
// Modified from the original: L3 raised from 0 to 70 so sustained sounds
// don't decay to silence, and R3 slowed slightly for a smoother sustain.
var epiano = []byte{
	95, 29, 20, 50, 99, 95, 70, 0, 0, 0, 0, 0, 0, 24, 79, 2, 0,
	95, 20, 20, 50, 99, 95, 70, 0, 0, 0, 0, 0, 0, 0, 99, 2, 0,
	95, 29, 20, 50, 99, 95, 70, 0, 0, 0, 0, 0, 0, 24, 89, 2, 0,
	95, 20, 20, 50, 99, 95, 70, 0, 0, 0, 0, 0, 0, 8, 99, 2, 0,
	95, 50, 35, 78, 99, 75, 70, 0, 0, 0, 0, 0, 0, 28, 58, 28, 0,
	96, 25, 25, 67, 99, 75, 70, 0, 0, 0, 0, 0, 0, 8, 99, 2, 0,

	99, 99, 99, 99, 50, 50, 50, 50, 4, 6, 34, 33, 0, 0, 56, 12,
	69, 46, 80, 73, 65, 78, 79, 32, 49, 32,
}

// OpParams holds the tunable parameters for a single DX7 operator.
type OpParams struct {
	Mode        uint8 // 0=ratio, 1=fixed
	FreqCoarse  uint8 // 0-31
	FreqFine    uint8 // 0-99
	OutputLevel uint8 // 0-99
	Rate1       uint8 // Attack Rate 0-99
	Rate2       uint8 // Decay Rate 0-99
	Level3      uint8 // Sustain Level 0-99
	Rate4       uint8 // Release Rate 0-99
}

const NumCandidateParams = 50

// Candidate represents a set of DX7 parameters to evaluate.
type Candidate struct {
	Algorithm uint8 // 0-31
	Ops       [6]OpParams
	Feedback  uint8  // 0-7
	RawBase   []byte // Optional 128-byte base template (e.g. from preset codebook)
	Fitness   float64 // Cached spectral distance (math.MaxFloat64 = unevaluated)

	// Pitch Envelope Generator (Pitch EG)
	PitchRate1  uint8 // 0-99 (default 99)
	PitchLevel1 uint8 // 0-99 (default 50 = neutral pitch)
	PitchLevel2 uint8 // 0-99 (default 50 = neutral pitch)

	// Direction memory & momentum weight per parameter (50 total)
	// Index 0: Algorithm, 1-48: Op1..Op6 (8 params per op), 49: Feedback
	Direction [NumCandidateParams]int8    // +1 or -1 direction memory
	Velocity  [NumCandidateParams]float64 // Selection weight / momentum multiplier
}

// RandomCandidate generates a random candidate.
func RandomCandidate(rng *rand.Rand) *Candidate {
	c := &Candidate{
		Algorithm:   uint8(rng.Intn(32)),
		Feedback:    uint8(rng.Intn(8)),
		Fitness:     math.MaxFloat64,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 6; i++ {
		c.Ops[i] = OpParams{
			Mode:        uint8(rng.Intn(2)),
			FreqCoarse:  uint8(rng.Intn(32)),
			FreqFine:    uint8(rng.Intn(100)),
			OutputLevel: 80 + uint8(rng.Intn(20)), // Output level 80-99
			Rate1:       50 + uint8(rng.Intn(50)), // 50-99
			Rate2:       20 + uint8(rng.Intn(60)), // 20-79
			Level3:      70 + uint8(rng.Intn(30)), // 70-99
			Rate4:       30 + uint8(rng.Intn(50)), // 30-79
		}
	}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		if rng.Intn(2) == 0 {
			c.Direction[i] = -1
		}
		c.Velocity[i] = 1.0
	}
	return c
}

// Copy returns a deep copy.
func Copy(c *Candidate) *Candidate {
	return c.Copy()
}

func (c *Candidate) Copy() *Candidate {
	d := &Candidate{
		Algorithm:   c.Algorithm,
		Feedback:    c.Feedback,
		Fitness:     c.Fitness,
		PitchRate1:  c.PitchRate1,
		PitchLevel1: c.PitchLevel1,
		PitchLevel2: c.PitchLevel2,
	}
	d.Ops = c.Ops
	d.Direction = c.Direction
	d.Velocity = c.Velocity
	if len(c.RawBase) == 128 {
		d.RawBase = make([]byte, 128)
		copy(d.RawBase, c.RawBase)
	}
	return d
}

// Encode packs the candidate into a 128-byte DX7 bulk patch.
// Non-tuned parameters are filled from RawBase template (or epiano defaults).
func (c *Candidate) Encode() []byte {
	patch := make([]byte, 128)
	if len(c.RawBase) == 128 {
		copy(patch, c.RawBase)
	} else {
		copy(patch, epiano)
	}

	for op := 0; op < 6; op++ {
		base := op * 17
		opP := c.Ops[op]

		// Bytes 0-3: Envelope Rates (R1, R2, R3, R4)
		patch[base+0] = opP.Rate1
		patch[base+1] = opP.Rate2
		patch[base+3] = opP.Rate4

		// Bytes 4-7: Envelope Levels (L1, L2, L3, L4)
		patch[base+6] = opP.Level3

		// For percussive envelopes (Level3 == 0), override L2=0 and R3=R2 even if RawBase is present
		if opP.Level3 == 0 {
			patch[base+2] = opP.Rate2 // R3 = R2
			patch[base+5] = 0         // L2 = 0 for instant percussive decay
		}

		// Fill default envelope stages if no RawBase template was provided
		if len(c.RawBase) != 128 {
			if opP.Level3 != 0 {
				patch[base+2] = opP.Rate2  // R3 = R2
				patch[base+5] = opP.Level3 // L2 = L3
			}
			patch[base+4] = 99          // L1 = max attack level
			patch[base+7] = 0           // L4 = 0 after release
		}

		// Byte 14: output level
		patch[base+14] = opP.OutputLevel

		// Byte 15: freq_coarse[6:1] | osc_mode[0]
		patch[base+15] = (opP.FreqCoarse << 1) | (opP.Mode & 1)

		// Byte 16: freq fine
		patch[base+16] = opP.FreqFine
	}

	// Byte 110: algorithm
	patch[110] = c.Algorithm

	// Byte 111: feedback[2:0] | osc_sync[bit 3 = 1]
	patch[111] = (patch[111] & 0xF8) | (c.Feedback & 0x07) | 0x08

	// Bytes 102-109: Pitch Envelope Generator (PR1..PR4, PL1..PL4)
	pRate1 := c.PitchRate1
	if pRate1 == 0 {
		pRate1 = 99
	}
	pLvl1 := c.PitchLevel1
	pLvl2 := c.PitchLevel2
	if pLvl1 == 0 && pLvl2 == 0 {
		pLvl1 = 50
		pLvl2 = 50
	}
	patch[102] = pRate1
	patch[106] = pLvl1
	patch[107] = pLvl2
	patch[108] = pLvl2

	return patch
}

// algOps defines the operator flags for all 32 DX7 algorithms.
// Mirroring the C++ MSFA synth engine definitions.
var algOps = [32][6]byte{
	{0xc1, 0x11, 0x11, 0x14, 0x01, 0x14}, // 1
	{0x01, 0x11, 0x11, 0x14, 0xc1, 0x14}, // 2
	{0xc1, 0x11, 0x14, 0x01, 0x11, 0x14}, // 3
	{0x41, 0x11, 0x94, 0x01, 0x11, 0x14}, // 4
	{0xc1, 0x14, 0x01, 0x14, 0x01, 0x14}, // 5
	{0x41, 0x94, 0x01, 0x14, 0x01, 0x14}, // 6
	{0xc1, 0x11, 0x05, 0x14, 0x01, 0x14}, // 7
	{0x01, 0x11, 0xc5, 0x14, 0x01, 0x14}, // 8
	{0x01, 0x11, 0x05, 0x14, 0xc1, 0x14}, // 9
	{0x01, 0x05, 0x14, 0xc1, 0x11, 0x14}, // 10
	{0xc1, 0x05, 0x14, 0x01, 0x11, 0x14}, // 11
	{0x01, 0x05, 0x05, 0x14, 0xc1, 0x14}, // 12
	{0xc1, 0x05, 0x05, 0x14, 0x01, 0x14}, // 13
	{0xc1, 0x05, 0x11, 0x14, 0x01, 0x14}, // 14
	{0x01, 0x05, 0x11, 0x14, 0xc1, 0x14}, // 15
	{0xc1, 0x11, 0x02, 0x25, 0x05, 0x14}, // 16
	{0x01, 0x11, 0x02, 0x25, 0xc5, 0x14}, // 17
	{0x01, 0x11, 0x11, 0xc5, 0x05, 0x14}, // 18
	{0xc1, 0x14, 0x14, 0x01, 0x11, 0x14}, // 19
	{0x01, 0x05, 0x14, 0xc1, 0x14, 0x14}, // 20
	{0x01, 0x14, 0x14, 0xc1, 0x14, 0x14}, // 21
	{0xc1, 0x14, 0x14, 0x14, 0x01, 0x14}, // 22
	{0xc1, 0x14, 0x14, 0x01, 0x14, 0x04}, // 23
	{0xc1, 0x14, 0x14, 0x14, 0x04, 0x04}, // 24
	{0xc1, 0x14, 0x14, 0x04, 0x04, 0x04}, // 25
	{0xc1, 0x05, 0x14, 0x01, 0x14, 0x04}, // 26
	{0x01, 0x05, 0x14, 0xc1, 0x14, 0x04}, // 27
	{0x04, 0xc1, 0x11, 0x14, 0x01, 0x14}, // 28
	{0xc1, 0x14, 0x01, 0x14, 0x04, 0x04}, // 29
	{0x04, 0xc1, 0x11, 0x14, 0x04, 0x04}, // 30
	{0xc1, 0x14, 0x04, 0x04, 0x04, 0x04}, // 31
	{0xc4, 0x04, 0x04, 0x04, 0x04, 0x04}, // 32
}

// IsCarrier returns true if the operator is an output carrier for the given algorithm.
func IsCarrier(algo uint8, opIdx int) bool {
	if algo >= 32 || opIdx < 0 || opIdx >= 6 {
		return false
	}
	return (algOps[algo][opIdx] & 3) == 0
}

// MutateInformed applies a mutation biased by velocity, acoustic guidance, and distance threshold.
// Returns the child, the parameter index mutated, and the delta applied.
func (c *Candidate) MutateInformed(rng *rand.Rand, targetF, candF TargetFeatures, bestDist float64) (*Candidate, int, int8) {
	d := c.Copy()

	// Adjust selection weights based on acoustic guidance
	weights := make([]float64, NumCandidateParams)
	copy(weights, d.Velocity[:])

	// Brightness guidance: if candidate is too dark, boost modulator levels and feedback
	if targetF.BrightnessScore > candF.BrightnessScore+0.1 {
		// Boost feedback selection weight
		weights[NumCandidateParams-1] *= 1.5
		// Boost modulator levels selection weight
		for i := 0; i < 6; i++ {
			if !IsCarrier(d.Algorithm, i) {
				weights[1+i*8+3] *= 1.5 // OutputLevel
			}
		}
	}

	// Weighted selection of parameter to mutate
	totalWeight := 0.0
	for _, w := range weights {
		totalWeight += w
	}
	r := rng.Float64() * totalWeight
	which := 0
	curr := 0.0
	for i, w := range weights {
		curr += w
		if r <= curr {
			which = i
			break
		}
	}

	// Occasionally randomize direction even if we have momentum (10% jitter)
	delta := d.Direction[which]
	if rng.Float64() < 0.1 {
		delta = -delta
	}

	// Step size thresholding: high distance -> larger steps
	magnitude := 1
	if bestDist > 1.0 {
		magnitude = 2 + rng.Intn(4) // Jumps of 2..5
	} else if bestDist > 0.6 {
		if d.Velocity[which] > 4.0 {
			magnitude = 1 + rng.Intn(2)
		}
	} else {
		magnitude = 1 // Fine-tuning micro-steps
	}
	step := delta * int8(magnitude)

	switch {
	case which == 0:
		d.Algorithm = uint8(rng.Intn(32))
	case which == NumCandidateParams-1:
		d.Feedback = clampU8(d.Feedback, step, 0, 7)
	default:
		opIdx := (which - 1) / 8
		paramIdx := (which - 1) % 8
		switch paramIdx {
		case 0:
			d.Ops[opIdx].Mode ^= 1
		case 1:
			d.Ops[opIdx].FreqCoarse = clampU8(d.Ops[opIdx].FreqCoarse, step, 0, 31)
		case 2:
			d.Ops[opIdx].FreqFine = clampU8(d.Ops[opIdx].FreqFine, step, 0, 99)
		case 3:
			d.Ops[opIdx].OutputLevel = clampU8(d.Ops[opIdx].OutputLevel, step, 75, 99)
		case 4:
			d.Ops[opIdx].Rate1 = clampU8(d.Ops[opIdx].Rate1, step, 10, 99)
		case 5:
			d.Ops[opIdx].Rate2 = clampU8(d.Ops[opIdx].Rate2, step, 5, 99)
		case 6:
			minL3 := uint8(0)
			if IsCarrier(d.Algorithm, opIdx) {
				minL3 = 50 // Carriers keep audible sustain floor (-36dB max attenuation)
			}
			d.Ops[opIdx].Level3 = clampU8(d.Ops[opIdx].Level3, step, minL3, 99)
		case 7:
			d.Ops[opIdx].Rate4 = clampU8(d.Ops[opIdx].Rate4, step, 10, 99)
		}
	}

	return d, which, step
}

// Mutate is the old random mutation, kept for compatibility if needed.
func (c *Candidate) Mutate(rng *rand.Rand) *Candidate {
	d, _, _ := c.MutateInformed(rng, TargetFeatures{}, TargetFeatures{}, 1.0)
	return d
}

// Crossover combines two parents to produce a child via uniform crossover.
func Crossover(a, b *Candidate, rng *rand.Rand) *Candidate {
	child := &Candidate{
		Fitness: math.MaxFloat64, // Mark as unevaluated
	}

	if rng.Intn(2) == 0 {
		child.Algorithm = a.Algorithm
		child.Direction[0] = a.Direction[0]
		child.Velocity[0] = a.Velocity[0]
	} else {
		child.Algorithm = b.Algorithm
		child.Direction[0] = b.Direction[0]
		child.Velocity[0] = b.Velocity[0]
	}

	if rng.Intn(2) == 0 {
		child.Feedback = a.Feedback
		child.Direction[NumCandidateParams-1] = a.Direction[NumCandidateParams-1]
		child.Velocity[NumCandidateParams-1] = a.Velocity[NumCandidateParams-1]
	} else {
		child.Feedback = b.Feedback
		child.Direction[NumCandidateParams-1] = b.Direction[NumCandidateParams-1]
		child.Velocity[NumCandidateParams-1] = b.Velocity[NumCandidateParams-1]
	}

	for i := 0; i < 6; i++ {
		idxBase := 1 + i*8

		if rng.Intn(2) == 0 {
			child.Ops[i] = a.Ops[i]
			for k := 0; k < 8; k++ {
				child.Direction[idxBase+k] = a.Direction[idxBase+k]
				child.Velocity[idxBase+k] = a.Velocity[idxBase+k]
			}
		} else {
			child.Ops[i] = b.Ops[i]
			for k := 0; k < 8; k++ {
				child.Direction[idxBase+k] = b.Direction[idxBase+k]
				child.Velocity[idxBase+k] = b.Velocity[idxBase+k]
			}
		}
	}

	return child
}

func randInt8(rng *rand.Rand, n int) int8 {
	return int8(rng.Intn(n) - n/2)
}

func clampU8(val uint8, delta int8, minVal, maxVal uint8) uint8 {
	v := int(val) + int(delta)
	if v < int(minVal) {
		return minVal
	}
	if v > int(maxVal) {
		return maxVal
	}
	return uint8(v)
}

func (c *Candidate) String() string {
	s := "algo=" + itoa(int(c.Algorithm)) + " fb=" + itoa(int(c.Feedback)) + "\n"
	for i, op := range c.Ops {
		s += "  op" + itoa(i+1) +
			": mode=" + itoa(int(op.Mode)) +
			" coarse=" + itoa(int(op.FreqCoarse)) +
			" fine=" + itoa(int(op.FreqFine)) +
			" level=" + itoa(int(op.OutputLevel)) + "\n"
	}
	return s
}

// SysexBulkDump creates a DX7 SysEx bulk dump (4104 bytes) with the candidate
// in voice slot 0 and default epiano in the remaining 31 slots.
// Format: F0 43 00 09 20 00 [4096 bytes voice data] [checksum] F7
func (c *Candidate) SysexBulkDump() []byte {
	return SysexBulkDumpMultiple([]*Candidate{c})
}

// SysexBulkDumpMultiple creates a DX7 SysEx bulk dump with multiple candidates
// in sequential voice slots. Remaining slots are filled with epiano defaults.
func SysexBulkDumpMultiple(candidates []*Candidate) []byte {
	bank := make([]byte, 4096)

	// Copy epiano defaults for all32 slots
	for i := 0; i < 32; i++ {
		copy(bank[i*128:], epiano)
	}

	// Overwrite slots with candidates and set custom names
	for i, c := range candidates {
		if i >= 32 {
			break
		}
		voiceData := c.Encode()
		// Set voice name e.g. "MATCH 01   "
		nameStr := fmt.Sprintf("MATCH %-2d  ", i+1)
		copy(voiceData[118:128], []byte(nameStr[:10]))
		copy(bank[i*128:], voiceData)
	}

	// Compute checksum: sum of all4096 data bytes, mod 128, subtracted from 128
	sum := 0
	for _, b := range bank {
		sum += int(b)
	}
	checksum := byte((128 - (sum % 128)) % 128)

	// Assemble the SysEx message
	syx := make([]byte, 0, 4104)
	syx = append(syx, 0xF0)       // Sysex start
	syx = append(syx, 0x43)       // Yamaha
	syx = append(syx, 0x00)       // Generic MIDI
	syx = append(syx, 0x09)       // DX7
	syx = append(syx, 0x20)       // Bulk dump
	syx = append(syx, 0x00)       // Voice
	syx = append(syx, bank...)    // 4096 bytes voice data
	syx = append(syx, checksum)   // Checksum
	syx = append(syx, 0xF7)       // Sysex end

	return syx
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	buf := [10]byte{}
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
