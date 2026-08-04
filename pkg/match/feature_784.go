package match

// Feature784Dim is the total dimensionality of the combined feature vector:
// 783 from ToExtendedVector() + 1 MIDI note = 784
const Feature784Dim = 783 + 1

// ExtractFeatures784 extracts the combined 784-dim feature vector for the patch classifier.
// Layout: [256 static | 512 temporal slices | 8 energy | 7 flux | 1 MIDI note]
// Uses the same extraction pipeline as the inference path for perfect consistency.
func ExtractFeatures784(samples []float64, sampleRate int, midiNote int) [Feature784Dim]float64 {
	var features [Feature784Dim]float64

	temporal := ExtractTemporalTimbreSpec(samples, sampleRate)
	ext := temporal.ToExtendedVector()
	copy(features[:783], ext)

	// MIDI note normalized to [0, 1]
	features[783] = float64(midiNote) / 127.0

	return features
}
