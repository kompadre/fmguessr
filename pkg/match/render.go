package match

import (
	"time"

	"github.com/kompadre/fmguessr/external/msfa"
)

// RenderSampleRate is the fixed sample rate used for rendering.
const RenderSampleRate = 44100

// MaxParallel is the number of channels MSFA supports (16).
const MaxParallel = 16

// RenderCandidate creates a fresh MSFA synth, loads the candidate patch,
// triggers a MIDI note, and renders audio for the given duration.
// Returns mono float64 samples normalized to [-1, 1].
func RenderCandidate(c *Candidate, noteNum uint8, duration time.Duration) []float64 {
	synth := msfa.NewSynth(float64(RenderSampleRate))
	defer synth.Free()

	patch := c.Encode()
	synth.SetVoice(0, patch)

	synth.WriteMidi([]byte{0x90, noteNum, 100})

	nSamples := int(float64(RenderSampleRate) * duration.Seconds())

	pcm := make([]int16, nSamples)
	synth.GetSamples(pcm)

	synth.WriteMidi([]byte{0x80, noteNum, 0})

	drainSamples := RenderSampleRate / 4
	drain := make([]int16, drainSamples)
	synth.GetSamples(drain)

	total := make([]float64, nSamples+drainSamples)
	for i := 0; i < nSamples; i++ {
		total[i] = float64(pcm[i]) / 32768.0
	}
	for i := 0; i < drainSamples; i++ {
		total[nSamples+i] = float64(drain[i]) / 32768.0
	}

	return total
}

// RenderCandidateShort renders a candidate for a short duration (e.g. 300ms)
// for fast peak-window comparison. Same as RenderCandidate but with shorter
// sustain period and proportionally shorter drain tail.
func RenderCandidateShort(c *Candidate, noteNum uint8, duration time.Duration) []float64 {
	synth := msfa.NewSynth(float64(RenderSampleRate))
	defer synth.Free()

	patch := c.Encode()
	synth.SetVoice(0, patch)

	synth.WriteMidi([]byte{0x90, noteNum, 100})

	nSamples := int(float64(RenderSampleRate) * duration.Seconds())

	pcm := make([]int16, nSamples)
	synth.GetSamples(pcm)

	synth.WriteMidi([]byte{0x80, noteNum, 0})

	// Shorter drain tail proportional to sustain (min 64ms)
	drainMs := duration.Milliseconds() / 4
	if drainMs < 64 {
		drainMs = 64
	}
	drainSamples := int(float64(RenderSampleRate) * float64(drainMs) / 1000.0)
	drain := make([]int16, drainSamples)
	synth.GetSamples(drain)

	total := make([]float64, nSamples+drainSamples)
	for i := 0; i < nSamples; i++ {
		total[i] = float64(pcm[i]) / 32768.0
	}
	for i := 0; i < drainSamples; i++ {
		total[nSamples+i] = float64(drain[i]) / 32768.0
	}

	return total
}

// RenderCandidatesParallel renders up to 16 candidates simultaneously on
// separate MSFA channels. Returns per-channel float64 audio (sustain + 250ms
// release tail), normalized to [-1, 1].
func RenderCandidatesParallel(candidates []*Candidate, noteNum uint8, duration time.Duration) [][]float64 {
	n := len(candidates)
	if n > MaxParallel {
		n = MaxParallel
	}

	synth := msfa.NewSynth(float64(RenderSampleRate))
	defer synth.Free()

	// Load each candidate's patch into its own slot, route each channel to it
	for ch := 0; ch < n; ch++ {
		patch := candidates[ch].Encode()
		synth.SetVoice(ch, patch)
		synth.ChangeProgram(uint8(ch), uint8(ch))
	}

	// Send Note On to all active channels
	for ch := 0; ch < n; ch++ {
		synth.WriteMidi([]byte{0x90 | byte(ch), noteNum, 100})
	}

	sustainSamples := int(float64(RenderSampleRate) * duration.Seconds())
	drainSamples := RenderSampleRate / 4
	totalSamples := sustainSamples + drainSamples

	// Allocate per-channel output buffers (int32 pre-clip, pre-filter)
	chBufs := make([][]int32, n)
	for ch := 0; ch < n; ch++ {
		chBufs[ch] = make([]int32, totalSamples)
	}

	// 1. Render sustain period in ONE single GetSamples call (processes Note On)
	mixedSustain := make([]int16, sustainSamples)
	synth.GetSamples(mixedSustain)

	// Read per-channel sustain audio
	for ch := 0; ch < n; ch++ {
		synth.GetChannelSamples(ch, chBufs[ch][:sustainSamples])
	}

	// 2. Send Note Off to all active channels
	for ch := 0; ch < n; ch++ {
		synth.WriteMidi([]byte{0x80 | byte(ch), noteNum, 0})
	}

	// 3. Render release tail in ONE single GetSamples call (processes Note Off)
	mixedDrain := make([]int16, drainSamples)
	synth.GetSamples(mixedDrain)

	// Read per-channel release audio
	drainSlice := make([]int32, drainSamples)
	for ch := 0; ch < n; ch++ {
		synth.GetChannelSamples(ch, drainSlice)
		copy(chBufs[ch][sustainSamples:], drainSlice)
	}

	// Convert int32 pre-clip to float64 normalized
	// The int16 output path does: (raw >> 4) >> 9 = raw >> 13, then /32768 for float
	// So the correct scale for pre-filter int32 is 1/(2^4 * 2^9 * 32768) = 1/2^28
	scale := 1.0 / float64(1<<28)
	result := make([][]float64, n)
	for ch := 0; ch < n; ch++ {
		result[ch] = make([]float64, totalSamples)
		for i := 0; i < totalSamples; i++ {
			result[ch][i] = float64(chBufs[ch][i]) * scale
		}
	}

	return result
}

// RenderCandidateWithGate renders a candidate patch holding Note On for gateDur,
// then sending Note Off to trigger the release stage for the remainder of totalDur.
func RenderCandidateWithGate(c *Candidate, noteNum uint8, gateDur, totalDur time.Duration) []float64 {
	synth := msfa.NewSynth(float64(RenderSampleRate))
	defer synth.Free()

	patch := c.Encode()
	synth.SetVoice(0, patch)

	totalSamples := int(float64(RenderSampleRate) * totalDur.Seconds())
	gateSamples := int(float64(RenderSampleRate) * gateDur.Seconds())
	if gateSamples > totalSamples {
		gateSamples = totalSamples
	}
	minGate := RenderSampleRate / 100 // 10ms minimum gate
	if gateSamples < minGate {
		gateSamples = minGate
	}

	total := make([]float64, totalSamples+RenderSampleRate/4)

	// 1. Note On phase (held for gateSamples)
	synth.WriteMidi([]byte{0x90, noteNum, 100})
	gatePCM := make([]int16, gateSamples)
	synth.GetSamples(gatePCM)
	for i := 0; i < gateSamples; i++ {
		total[i] = float64(gatePCM[i]) / 32768.0
	}

	// 2. Note Off phase (release phase for remainder of totalDur)
	synth.WriteMidi([]byte{0x80, noteNum, 0})
	releaseSamples := totalSamples - gateSamples
	if releaseSamples > 0 {
		releasePCM := make([]int16, releaseSamples)
		synth.GetSamples(releasePCM)
		for i := 0; i < releaseSamples; i++ {
			total[gateSamples+i] = float64(releasePCM[i]) / 32768.0
		}
	}

	// 3. Extra drain tail (250ms)
	drainSamples := RenderSampleRate / 4
	drain := make([]int16, drainSamples)
	synth.GetSamples(drain)
	for i := 0; i < drainSamples; i++ {
		total[totalSamples+i] = float64(drain[i]) / 32768.0
	}

	return total
}
