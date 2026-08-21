package match

import (
	"encoding/base64"
	"fmt"
	"math"
	"os"
	"time"
)

// MorphConfig holds parameters for the patch morph operation.
type MorphConfig struct {
	Source  *Candidate
	Target  *Candidate
	Note    uint8
	Dur     time.Duration
	Ratio   float64 // target distance ratio (0.5 = halve distance)
	MaxIter int     // optimization iterations after lerp (0 = lerp only)
	Verbose bool
}

// MorphResult holds the outcome of a morph operation.
type MorphResult struct {
	Morphed     *Candidate
	SourceDist  float64
	MorphDist   float64
	TargetRatio float64
	Alpha       float64
	Iterations  int
}

// LerpCandidates interpolates between source and target at the given alpha (0..1).
// Discrete params (Algorithm, Feedback, Mode) are rounded to nearest integer.
// Continuous params are clamped to valid DX7 ranges.
func LerpCandidates(source, target *Candidate, alpha float64) *Candidate {
	c := source.Copy()
	inv := 1.0 - alpha

	// Algorithm: round to nearest
	c.Algorithm = uint8(math.Round(float64(source.Algorithm)*inv + float64(target.Algorithm)*alpha))

	// Feedback: round to nearest
	c.Feedback = uint8(math.Round(float64(source.Feedback)*inv + float64(target.Feedback)*alpha))

	// Pitch EG
	c.PitchRate1 = clampRoundU8(float64(source.PitchRate1)*inv+float64(target.PitchRate1)*alpha, 0, 99)
	c.PitchLevel1 = clampRoundU8(float64(source.PitchLevel1)*inv+float64(target.PitchLevel1)*alpha, 0, 99)
	c.PitchLevel2 = clampRoundU8(float64(source.PitchLevel2)*inv+float64(target.PitchLevel2)*alpha, 0, 99)

	// Per-operator params
	for i := 0; i < 6; i++ {
		s := source.Ops[i]
		t := target.Ops[i]

		// Mode: pick nearest
		if alpha < 0.5 {
			c.Ops[i].Mode = s.Mode
		} else {
			c.Ops[i].Mode = t.Mode
		}

		c.Ops[i].FreqCoarse = clampRoundU8(float64(s.FreqCoarse)*inv+float64(t.FreqCoarse)*alpha, 0, 31)
		c.Ops[i].FreqFine = clampRoundU8(float64(s.FreqFine)*inv+float64(t.FreqFine)*alpha, 0, 99)
		c.Ops[i].OutputLevel = clampRoundU8(float64(s.OutputLevel)*inv+float64(t.OutputLevel)*alpha, 0, 99)
		c.Ops[i].Rate1 = clampRoundU8(float64(s.Rate1)*inv+float64(t.Rate1)*alpha, 0, 99)
		c.Ops[i].Rate2 = clampRoundU8(float64(s.Rate2)*inv+float64(t.Rate2)*alpha, 0, 99)
		c.Ops[i].Level3 = clampRoundU8(float64(s.Level3)*inv+float64(t.Level3)*alpha, 0, 99)
		c.Ops[i].Rate4 = clampRoundU8(float64(s.Rate4)*inv+float64(t.Rate4)*alpha, 0, 99)
	}

	return c
}

func clampRoundU8(v float64, min, max uint8) uint8 {
	r := int(math.Round(v))
	if r < int(min) {
		return min
	}
	if r > int(max) {
		return max
	}
	return uint8(r)
}

// binarySearchLerp finds the alpha where spectral distance is closest to targetDist.
func binarySearchLerp(source, target *Candidate, targetAudio []float64, targetDist float64, note uint8, dur time.Duration) (float64, *Candidate) {
	return binarySearchLerpEx(source, target, targetAudio, targetDist, 0.0, 1.0, note, dur)
}

// binarySearchLerpEx is like binarySearchLerp but with custom alpha bounds.
func binarySearchLerpEx(source, target *Candidate, targetAudio []float64, targetDist float64, lo, hi float64, note uint8, dur time.Duration) (float64, *Candidate) {
	bestAlpha := (lo + hi) / 2.0
	bestDist := math.MaxFloat64
	var bestCand *Candidate

	for i := 0; i < 15; i++ {
		mid := (lo + hi) / 2.0
		cand := LerpCandidates(source, target, mid)
		candAudio := RenderCandidate(cand, note, dur)
		dist := MelCosineDistance(targetAudio, candAudio, RenderSampleRate)

		diff := math.Abs(dist - targetDist)
		if diff < bestDist {
			bestDist = diff
			bestAlpha = mid
			bestCand = cand
		}

		if dist > targetDist {
			hi = mid
		} else {
			lo = mid
		}
	}

	return bestAlpha, bestCand
}

// Morph interpolates source toward target. The ratio controls how far along
// the path from source to target the result sits (0.0 = source, 1.0 = target).
// Negative ratios extrapolate away from target (alpha < 0).
// The binary search finds the interpolation alpha, then optimization nudges
// individual parameters for finer adjustment.
func Morph(cfg MorphConfig) MorphResult {
	if cfg.Dur == 0 {
		cfg.Dur = time.Second
	}
	if cfg.Note == 0 {
		cfg.Note = 60
	}
	ratio := cfg.Ratio
	if ratio == 0 {
		ratio = 0.5
	}

	// Render source and target
	sourceAudio := RenderCandidate(cfg.Source, cfg.Note, cfg.Dur)
	targetAudio := RenderCandidate(cfg.Target, cfg.Note, cfg.Dur)
	sourceDist := MelCosineDistance(targetAudio, sourceAudio, RenderSampleRate)

	if cfg.Verbose {
		fmt.Printf("Source distance: %.6f\n", sourceDist)
		fmt.Printf("Target distance: %.6f (%.0f%% of source)\n", sourceDist*ratio, ratio*100)
	}

	// For negative ratios, search alpha in [-1, 0] range
	// For positive ratios, search alpha in [0, 1] range
	lo, hi := 0.0, 1.0
	if ratio < 0 {
		lo = -1.0
		hi = 0.0
	}
	targetDist := sourceDist * math.Abs(ratio)

	// Lerp pass: binary search for alpha that gets closest to target distance
	alpha, morphed := binarySearchLerpEx(cfg.Source, cfg.Target, targetAudio, targetDist, lo, hi, cfg.Note, cfg.Dur)

	morphedAudio := RenderCandidate(morphed, cfg.Note, cfg.Dur)
	morphDist := MelCosineDistance(targetAudio, morphedAudio, RenderSampleRate)

	if cfg.Verbose {
		fmt.Printf("After lerp (α=%.4f): distance=%.6f (ratio=%.4f)\n", alpha, morphDist, morphDist/sourceDist)
	}

	// Optimization pass: nudge parameters in desired direction
	// Each iteration modifies 2-3 knobs for bigger jumps toward target
	iterCount := 0
	if cfg.MaxIter > 0 {
		current := morphed.Copy()
		currentDist := morphDist

		for iter := 0; iter < cfg.MaxIter; iter++ {
			candidate := current.Copy()

			// Modify 2-3 knobs per iteration for faster convergence
			numKnobs := 2 + iter%2 // alternates between 2 and 3
			for k := 0; k < numKnobs; k++ {
				paramIdx := (iter*3 + k) % 50
				if ratio < 0 {
					nudgeParamAway(candidate, cfg.Target, paramIdx)
				} else {
					nudgeParamToward(candidate, cfg.Target, paramIdx)
				}
			}

			candidateAudio := RenderCandidate(candidate, cfg.Note, cfg.Dur)
			newDist := MelCosineDistance(targetAudio, candidateAudio, RenderSampleRate)

			if newDist < currentDist {
				current = candidate
				currentDist = newDist
				iterCount++

				if cfg.Verbose && iterCount%10 == 0 {
					fmt.Printf("  iter %d: distance=%.6f (ratio=%.4f)\n", iter, currentDist, currentDist/sourceDist)
				}
			}

			// Early exit if we've reached the target
			if currentDist <= targetDist {
				break
			}
		}

		morphed = current
		morphDist = currentDist
	}

	return MorphResult{
		Morphed:     morphed,
		SourceDist:  sourceDist,
		MorphDist:   morphDist,
		TargetRatio: morphDist / sourceDist,
		Alpha:       alpha,
		Iterations:  iterCount,
	}
}

// nudgeParamToward moves one parameter of c toward the corresponding parameter of target.
// Step size scales with distance — large gaps get bigger steps.
func nudgeParamToward(c, target *Candidate, paramIdx int) {
	if paramIdx == 0 {
		c.Algorithm = target.Algorithm
		return
	}
	if paramIdx == 49 {
		c.Feedback = target.Feedback
		return
	}

	opIdx := (paramIdx - 1) / 8
	paramLocal := (paramIdx - 1) % 8
	if opIdx >= 6 {
		return
	}

	switch paramLocal {
	case 0: // Mode
		c.Ops[opIdx].Mode = target.Ops[opIdx].Mode
	case 1: // FreqCoarse
		c.Ops[opIdx].FreqCoarse = nudgeU8(c.Ops[opIdx].FreqCoarse, target.Ops[opIdx].FreqCoarse, 0, 31)
	case 2: // FreqFine
		c.Ops[opIdx].FreqFine = nudgeU8(c.Ops[opIdx].FreqFine, target.Ops[opIdx].FreqFine, 0, 99)
	case 3: // OutputLevel
		c.Ops[opIdx].OutputLevel = nudgeU8(c.Ops[opIdx].OutputLevel, target.Ops[opIdx].OutputLevel, 0, 99)
	case 4: // Rate1
		c.Ops[opIdx].Rate1 = nudgeU8(c.Ops[opIdx].Rate1, target.Ops[opIdx].Rate1, 0, 99)
	case 5: // Rate2
		c.Ops[opIdx].Rate2 = nudgeU8(c.Ops[opIdx].Rate2, target.Ops[opIdx].Rate2, 0, 99)
	case 6: // Level3
		c.Ops[opIdx].Level3 = nudgeU8(c.Ops[opIdx].Level3, target.Ops[opIdx].Level3, 0, 99)
	case 7: // Rate4
		c.Ops[opIdx].Rate4 = nudgeU8(c.Ops[opIdx].Rate4, target.Ops[opIdx].Rate4, 0, 99)
	}
}

// nudgeU8 moves current by ±1 toward target, clamped to [min, max].
func nudgeU8(current, target, min, max uint8) uint8 {
	if current == target {
		return current
	}
	if current < target {
		v := int(current) + 1
		if v > int(max) {
			return max
		}
		return uint8(v)
	}
	v := int(current) - 1
	if v < int(min) {
		return min
	}
	return uint8(v)
}

func nudgeParamAway(c, target *Candidate, paramIdx int) {
	if paramIdx == 0 || paramIdx == 49 {
		return // Don't change algorithm or feedback when going away
	}

	opIdx := (paramIdx - 1) / 8
	paramLocal := (paramIdx - 1) % 8
	if opIdx >= 6 {
		return
	}

	switch paramLocal {
	case 0: // Mode — skip
	case 1: // FreqCoarse — nudge away from target
		c.Ops[opIdx].FreqCoarse = nudgeU8Away(c.Ops[opIdx].FreqCoarse, target.Ops[opIdx].FreqCoarse, 0, 31)
	case 2: // FreqFine
		c.Ops[opIdx].FreqFine = nudgeU8Away(c.Ops[opIdx].FreqFine, target.Ops[opIdx].FreqFine, 0, 99)
	case 3: // OutputLevel
		c.Ops[opIdx].OutputLevel = nudgeU8Away(c.Ops[opIdx].OutputLevel, target.Ops[opIdx].OutputLevel, 0, 99)
	case 4: // Rate1
		c.Ops[opIdx].Rate1 = nudgeU8Away(c.Ops[opIdx].Rate1, target.Ops[opIdx].Rate1, 0, 99)
	case 5: // Rate2
		c.Ops[opIdx].Rate2 = nudgeU8Away(c.Ops[opIdx].Rate2, target.Ops[opIdx].Rate2, 0, 99)
	case 6: // Level3
		c.Ops[opIdx].Level3 = nudgeU8Away(c.Ops[opIdx].Level3, target.Ops[opIdx].Level3, 0, 99)
	case 7: // Rate4
		c.Ops[opIdx].Rate4 = nudgeU8Away(c.Ops[opIdx].Rate4, target.Ops[opIdx].Rate4, 0, 99)
	}
}

func nudgeU8Away(current, target, min, max uint8) uint8 {
	if current == target {
		// At target — nudge in arbitrary direction
		if current < max {
			return current + 1
		}
		return current - 1
	}
	if current < target {
		// Below target — move away (decrease)
		v := int(current) - 1
		if v < int(min) {
			return min
		}
		return uint8(v)
	}
	// Above target — move away (increase)
	v := int(current) + 1
	if v > int(max) {
		return max
	}
	return uint8(v)
}

// LoadSyxPatch loads a .syx file and returns the first voice as a Candidate.
// Supports DX7 bulk dumps (4104 bytes, 32 voices) and single voice patches.
func LoadSyxPatch(path string) (*Candidate, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", path, err)
	}

	// DX7 bulk dump: F0 43 00 09 20 00 [4096 bytes] [checksum] F7 = 4104 bytes
	// Strip SysEx header (6 bytes) and checksum+EOX (2 bytes) → 4096 bytes
	if len(data) >= 4104 && data[0] == 0xF0 && data[1] == 0x43 {
		voiceData := data[6 : 6+4096] // 32 × 128-byte voices
		return DecodeVoicePatch(voiceData[0:128]), nil // slot 0 only
	}

	// Single voice: strip SysEx framing if present, take first 128 bytes
	if len(data) >= 136 && data[0] == 0xF0 && data[1] == 0x43 {
		// Find the voice data after SysEx header (typically at offset 6)
		return DecodeVoicePatch(data[6 : 6+128]), nil
	}

	// Raw 128-byte patch (no SysEx framing)
	if len(data) == 128 {
		return DecodeVoicePatch(data), nil
	}

	return nil, fmt.Errorf("unsupported .syx format (%d bytes)", len(data))
}

// DecodeBase64 decodes a base64-encoded string to bytes.
func DecodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}

// EncodeBase64 encodes bytes to a base64 string.
func EncodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}
