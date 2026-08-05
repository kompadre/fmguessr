package match

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// AudioEncoder wraps a 1D ResNet ONNX model for extracting sliding-window
// audio embeddings. The model maps (B, 1, winSamples) raw waveform → (B, embDim) L2-normalized embeddings.
type AudioEncoder struct {
	session    *ort.DynamicAdvancedSession
	embDim     int // output embedding dimension (e.g. 128)
	winSamples int // window size in samples (e.g. 11025 = 250ms at 44.1kHz)
	hopSamples int // stride in samples (e.g. 1323 = 30ms at 44.1kHz)
}

// NewAudioEncoder loads a 1D ResNet ONNX model for audio embedding extraction.
// The model must accept input shape (B, 1, winSamples) and produce (B, embDim).
func NewAudioEncoder(modelPath string) (*AudioEncoder, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("init ONNX env: %w", err)
		}
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"audio"}, []string{"embedding"}, nil)
	if err != nil {
		return nil, fmt.Errorf("load encoder model: %w", err)
	}

	// Default dimensions — these should be read from model metadata in production.
	// For now, hardcode to the training configuration.
	return &AudioEncoder{
		session:    session,
		embDim:     128,
		winSamples: 11025, // 250ms at 44.1kHz
		hopSamples: 1323,  // 30ms at 44.1kHz
	}, nil
}

// EmbedSequence extracts sliding-window embeddings from PCM audio.
// Returns one []float32 embedding per window position.
func (e *AudioEncoder) EmbedSequence(samples []float64) [][]float32 {
	if len(samples) < e.winSamples {
		return nil
	}

	// Count windows
	nWindows := (len(samples)-e.winSamples)/e.hopSamples + 1
	if nWindows <= 0 {
		return nil
	}

	// Slice into windows
	windows := make([][]float32, nWindows)
	for i := 0; i < nWindows; i++ {
		start := i * e.hopSamples
		end := start + e.winSamples
		if end > len(samples) {
			end = len(samples)
		}
		windows[i] = make([]float32, e.winSamples)
		for j := start; j < end; j++ {
			windows[i][j-start] = float32(samples[j])
		}
	}

	return e.EmbedBatch(windows)
}

// EmbedBatch runs batched ONNX inference on pre-sliced windows.
// Input: [][]float32 windows (each length winSamples).
// Output: [][]float32 embeddings (each length embDim).
func (e *AudioEncoder) EmbedBatch(windows [][]float32) [][]float32 {
	batchSize := len(windows)
	if batchSize == 0 {
		return nil
	}

	// Create input tensor: (batchSize, 1, winSamples)
	inputData := make([]float32, batchSize*e.winSamples)
	for i, w := range windows {
		copy(inputData[i*e.winSamples:], w)
	}

	inputTensor, err := ort.NewTensor[float32](
		ort.NewShape(int64(batchSize), 1, int64(e.winSamples)), inputData)
	if err != nil {
		return nil
	}
	defer inputTensor.Destroy()

	// Create output tensor placeholder
	outputData := make([]float32, batchSize*e.embDim)
	outputTensor, err := ort.NewTensor[float32](
		ort.NewShape(int64(batchSize), int64(e.embDim)), outputData)
	if err != nil {
		return nil
	}
	defer outputTensor.Destroy()

	// Run inference
	if err := e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return nil
	}

	// Extract embeddings
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		result[i] = make([]float32, e.embDim)
		copy(result[i], outputData[i*e.embDim:(i+1)*e.embDim])
	}

	return result
}

// EmbedSingle extracts a single embedding from a PCM clip.
// The clip must be at least winSamples long.
func (e *AudioEncoder) EmbedSingle(samples []float64) []float32 {
	if len(samples) < e.winSamples {
		return nil
	}
	window := make([]float32, e.winSamples)
	for i := 0; i < e.winSamples; i++ {
		window[i] = float32(samples[i])
	}
	result := e.EmbedBatch([][]float32{window})
	if len(result) == 0 {
		return nil
	}
	return result[0]
}

// EmbDim returns the embedding dimension.
func (e *AudioEncoder) EmbDim() int {
	return e.embDim
}

// WindowSamples returns the window size in samples.
func (e *AudioEncoder) WindowSamples() int {
	return e.winSamples
}

// HopSamples returns the stride in samples.
func (e *AudioEncoder) HopSamples() int {
	return e.hopSamples
}

// Close releases the ONNX session resources.
func (e *AudioEncoder) Close() {
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
}

// L2Normalize normalizes a float32 vector to unit length in-place.
func L2Normalize(v []float32) {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm < 1e-10 {
		return
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
}
