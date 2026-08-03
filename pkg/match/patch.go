package match

import (
	"fmt"
	"math"
	"math/rand"
	"time"
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

// DX7LabelDims is the number of normalized DX7 parameters in training labels.
// Layout: Algorithm(1) + 6 ops × 8 params(48) + Feedback(1) + PitchEnv(3) = 53
const DX7LabelDims = 53

// Candidate represents a set of DX7 parameters to evaluate.
type Candidate struct {
	Algorithm uint8 // 0-31
	Ops       [6]OpParams
	Feedback  uint8  // 0-7
	RawBase   []byte // Optional 128-byte base template (e.g. from preset codebook)
	Fitness   float64 // Cached spectral distance (math.MaxFloat64 = unevaluated)
	Name      string  // Voice name (max 10 chars, used in .syx export)

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
			minLevel := uint8(75)
			if IsCarrier(d.Algorithm, opIdx) {
				minLevel = 85 // Carriers maintain minimum output level floor
			}
			d.Ops[opIdx].OutputLevel = clampU8(d.Ops[opIdx].OutputLevel, step, minLevel, 99)
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
		// Use candidate name if set, otherwise default
		name := c.Name
		if name == "" {
			name = fmt.Sprintf("MATCH %-2d  ", i+1)
		}
		copy(voiceData[118:128], []byte(name[:10]))
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

// SanitizeVoiceName converts a human-readable name into a valid DX7 voice name:
// uppercase, alphanumeric ASCII only, truncated to 10 characters.
func SanitizeVoiceName(raw string) string {
	var buf [10]byte
	n := 0
	for _, r := range raw {
		if n >= 10 {
			break
		}
		if r >= 'a' && r <= 'z' {
			buf[n] = byte(r - 32) // uppercase
			n++
		} else if r >= 'A' && r <= 'Z' {
			buf[n] = byte(r)
			n++
		} else if r >= '0' && r <= '9' {
			buf[n] = byte(r)
			n++
		}
	}
	// Pad with spaces
	for n < 10 {
		buf[n] = ' '
		n++
	}
	return string(buf[:10])
}

// DeduplicateVoiceNames ensures all names in the slice are unique by appending
// a serial number (01-99) to duplicates. Names are already assumed sanitized.
func DeduplicateVoiceNames(names []string) []string {
	seen := make(map[string]int, len(names))
	result := make([]string, len(names))
	for i, name := range names {
		key := name
		if count, exists := seen[key]; exists {
			seen[key] = count + 1
			// Truncate base to 7 chars, append " XX" serial
			base := name[:7]
			result[i] = fmt.Sprintf("%s %02d", base, count+1)
		} else {
			seen[key] = 1
			result[i] = name
		}
	}
	return result
}

// MorphName generates a morph patch name from two parent names.
// Uses first letters of each parent separated by underscore (e.g. "S_P").
// Falls back to numbered names if initials collide.
func MorphName(nameA, nameB string) string {
	a := "?"
	b := "?"
	for _, r := range nameA {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			a = string(r)
			break
		}
	}
	for _, r := range nameB {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b = string(r)
			break
		}
	}
	return a + "_" + b
}

// Pure Waveform Shape Base Templates

// SineBase returns a clean pure sine wave candidate.
func SineBase() *Candidate {
	c := &Candidate{
		Algorithm:   0, // Algo 1: Op 1 is carrier
		Feedback:    0,
		Fitness:     math.MaxFloat64,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 5; i++ {
		c.Ops[i] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 0, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	}
	c.Ops[5] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 99, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}
	return c
}

// SquareBase returns a clean pure square wave candidate (1:2 FM ratio).
func SquareBase() *Candidate {
	c := &Candidate{
		Algorithm:   0, // Algo 1: Op 2 modulates Op 1
		Feedback:    0,
		Fitness:     math.MaxFloat64,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 4; i++ {
		c.Ops[i] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 0, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	}
	c.Ops[4] = OpParams{Mode: 0, FreqCoarse: 2, FreqFine: 0, OutputLevel: 78, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	c.Ops[5] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 99, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}
	return c
}

// SawtoothBase returns a clean pure sawtooth wave candidate (1:1 FM ratio with feedback).
func SawtoothBase() *Candidate {
	c := &Candidate{
		Algorithm:   0, // Algo 1: Op 2 modulates Op 1
		Feedback:    6,
		Fitness:     math.MaxFloat64,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 4; i++ {
		c.Ops[i] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 0, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	}
	c.Ops[4] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 82, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	c.Ops[5] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 99, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}
	return c
}

// TriangleBase returns a clean pure triangle wave candidate (soft 1:2 FM ratio).
func TriangleBase() *Candidate {
	c := &Candidate{
		Algorithm:   0, // Algo 1
		Feedback:    0,
		Fitness:     math.MaxFloat64,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 4; i++ {
		c.Ops[i] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 0, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	}
	c.Ops[4] = OpParams{Mode: 0, FreqCoarse: 2, FreqFine: 0, OutputLevel: 58, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	c.Ops[5] = OpParams{Mode: 0, FreqCoarse: 1, FreqFine: 0, OutputLevel: 99, Rate1: 99, Rate2: 99, Level3: 99, Rate4: 99}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}
	return c
}

// DefaultWaveformSeeds returns all 5 foundational base seeds: Sine, Square, Sawtooth, Triangle, and Piano1.
func DefaultWaveformSeeds() []*Candidate {
	return []*Candidate{
		SineBase(),
		SquareBase(),
		SawtoothBase(),
		TriangleBase(),
		DecodeVoicePatch(epiano),
	}
}

// InitVoice returns a clean "init" patch with all operators silent, LFO off,
// pitch envelope flat, and no keyboard scaling. Used as the starting point for
// constructive beam search so there are no hidden biases from existing presets.
func InitVoice() *Candidate {
	base := make([]byte, 128)

	// Fill all 6 operator blocks (17 bytes each) with silent defaults
	for op := 0; op < 6; op++ {
		off := op * 17
		base[off+0] = 99  // R1: instant attack
		base[off+1] = 99  // R2: instant decay
		base[off+2] = 99  // R3 = R2
		base[off+3] = 0   // R4: instant release
		base[off+4] = 99  // L1: max attack level
		base[off+5] = 0   // L2: zero (Level3 == 0 → percussive)
		base[off+6] = 0   // L3: zero sustain → silent
		base[off+7] = 0   // L4: zero after release
		base[off+14] = 0  // OutputLevel: silent
		base[off+15] = 2  // FreqCoarse=1, Mode=0 (ratio)
		base[off+16] = 0  // FreqFine: zero
	}

	// LFO: all zeros (off)
	// Pitch EG: flat at 50 (bytes 102-109)
	base[102] = 99 // PR1
	base[103] = 0  // PR2
	base[104] = 0  // PR3
	base[105] = 0  // PR4
	base[106] = 50 // PL1: neutral
	base[107] = 50 // PL2: neutral
	base[108] = 50 // PL3: neutral
	base[109] = 50 // PL4: neutral

	// Algorithm = 31 (Algo 32), Feedback = 0
	base[110] = 31
	base[111] = 0x08 // feedback=0, osc_sync=1

	// Keyboard scaling: flat (no scaling)
	for i := 102; i < 110; i++ {
		// Already set above
	}

	c := &Candidate{
		Algorithm:   31, // Algo 32
		Fitness:     math.MaxFloat64,
		RawBase:     base,
		PitchRate1:  99,
		PitchLevel1: 50,
		PitchLevel2: 50,
	}
	for i := 0; i < 6; i++ {
		c.Ops[i] = OpParams{OutputLevel: 0, Level3: 0, Rate1: 99, Rate2: 99, Rate4: 0}
	}
	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}
	return c
}

// MuteOperator silences an operator by zeroing its output level and setting
// its envelope to instant attack → instant decay → zero sustain.
func MuteOperator(c *Candidate, opIdx int) {
	c.Ops[opIdx].OutputLevel = 0
	c.Ops[opIdx].Level3 = 0
	c.Ops[opIdx].Rate1 = 99
	c.Ops[opIdx].Rate2 = 99
	c.Ops[opIdx].Rate4 = 0
}

// CalibrateCarrierLevels boosts carrier operator OutputLevel settings so that the candidate's
// native rendered RMS matches rawTargetRMS.
func (c *Candidate) CalibrateCarrierLevels(rawTargetRMS float64, note uint8, dur time.Duration) {
	if rawTargetRMS < 1e-4 {
		return
	}
	audio := RenderCandidate(c, note, dur)
	candRMS := CalcRMS(audio)
	if candRMS < 1e-6 {
		return
	}

	gainRatio := rawTargetRMS / candRMS
	if gainRatio > 1.05 {
		dbBoost := 20.0 * math.Log10(gainRatio)
		levelBoost := int(math.Round(dbBoost / 0.75))
		if levelBoost > 0 {
			for op := 0; op < 6; op++ {
				if IsCarrier(c.Algorithm, op) {
					newLvl := int(c.Ops[op].OutputLevel) + levelBoost
					if newLvl > 99 {
						newLvl = 99
					}
					c.Ops[op].OutputLevel = uint8(newLvl)
				}
			}
		}
	}
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
