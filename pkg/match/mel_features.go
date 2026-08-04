package match

// MelSpecSize is the side length of the square mel spectrogram.
const MelSpecSize = 128

// MelSpecFeatureDim is the flattened mel spectrogram dimension (128 * 128).
const MelSpecFeatureDim = MelSpecSize * MelSpecSize

// ExtractMelSpectrogram extracts a 128×128 mel spectrogram from rendered audio
// and flattens it to a 1D feature vector for the CNN classifier.
func ExtractMelSpectrogram(samples []float64, sampleRate int) [MelSpecFeatureDim]float64 {
	mel := ComputeMelSpectrogram(samples, sampleRate)
	var flat [MelSpecFeatureDim]float64
	for i := 0; i < MelSpecSize; i++ {
		for j := 0; j < MelSpecSize; j++ {
			flat[i*MelSpecSize+j] = mel[i][j]
		}
	}
	return flat
}
