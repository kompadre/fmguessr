package match

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// PatchClassifier runs the patch classification NN model and returns
// a 128-byte DX7 patch directly from embedded model data.
type PatchClassifier struct {
	session       *ort.AdvancedSession
	inputTensor   *ort.Tensor[float32]
	midiTensor    *ort.Tensor[float32]   // 37 logits
	patchTensor   *ort.Tensor[float32]   // N logits
	patchBytes    []byte                 // N*128 bytes embedded in model
	numPatches    int
	noteMin       int
	noteMax       int
	featMin       [Feature784Dim]float64 // normalization stats
	featMax       [Feature784Dim]float64
}

// PatchPrediction holds the classifier output.
type PatchPrediction struct {
	PatchBytes []byte  // 128-byte DX7 voice patch
	MidiKey    int     // Predicted MIDI key (absolute, e.g. 48=C3)
	Confidence float64 // Softmax confidence for the winning patch
	TopK       []PatchMatch // Top-K predictions
}

// PatchMatch is one entry in the top-K predictions.
type PatchMatch struct {
	Index      int
	Confidence float64
	PatchBytes []byte
}

// NewPatchClassifier loads an ONNX model with embedded patch bytes.
// Looks for the model file and companion .norm.npy file for normalization stats.
func NewPatchClassifier(modelPath string) (*PatchClassifier, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ONNX env: %w", err)
		}
	}

	// Load patch bytes from companion .patches.npy to determine N
	patchBytes, numPatches, err := loadPatchBytes(modelPath + ".patches.npy")
	if err != nil {
		return nil, fmt.Errorf("failed to load patch bytes: %w", err)
	}

	inputNames := []string{"features"}
	outputNames := []string{"midi_logits", "patch_logits"}

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, Feature784Dim))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	midiTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, MIDIKeyClasses))
	if err != nil {
		return nil, fmt.Errorf("failed to create midi output tensor: %w", err)
	}

	patchTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(numPatches)))
	if err != nil {
		return nil, fmt.Errorf("failed to create patch output tensor: %w", err)
	}

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{midiTensor, patchTensor}

	p, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		inputs, outputs, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load classifier ONNX session: %w", err)
	}

	pc := &PatchClassifier{
		session:     p,
		inputTensor: inputTensor,
		midiTensor:  midiTensor,
		patchTensor: patchTensor,
		patchBytes:  patchBytes,
		numPatches:  numPatches,
	}

	// Load normalization stats from companion file
	if err := pc.loadNormStats(modelPath + ".norm.bin"); err != nil {
		log.Printf("Warning: could not load normalization stats: %v (using identity)", err)
		for i := 0; i < Feature784Dim; i++ {
			pc.featMin[i] = 0
			pc.featMax[i] = 1
		}
	}

	fmt.Printf("[classifier] Loaded patch classifier: %s (%d patches)\n", modelPath, numPatches)
	return pc, nil
}

// loadPatchBytes reads patch bytes from a .npy file saved by the training script.
// Format: numpy array of shape (N, 128) with dtype uint8.
func loadPatchBytes(path string) ([]byte, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("cannot open %s: %w", path, err)
	}
	defer f.Close()

	// Read npy header: magic(6) + major(1) + minor(1) + header_len(2) + header_str
	header := make([]byte, 6)
	if _, err := f.Read(header); err != nil {
		return nil, 0, fmt.Errorf("read npy magic: %w", err)
	}
	if string(header[:6]) != "\x93NUMPY" {
		return nil, 0, fmt.Errorf("not a valid npy file")
	}

	// Read version
	ver := make([]byte, 2)
	if _, err := f.Read(ver); err != nil {
		return nil, 0, err
	}

	// Read header length (2 bytes for v1, 4 bytes for v2+)
	var headerLen int
	if ver[0] == 1 {
		bl := make([]byte, 2)
		f.Read(bl)
		headerLen = int(bl[0]) | int(bl[1])<<8
	} else {
		bl := make([]byte, 4)
		f.Read(bl)
		headerLen = int(bl[0]) | int(bl[1])<<8 | int(bl[2])<<16 | int(bl[3])<<24
	}

	// Read header string
	headerStr := make([]byte, headerLen)
	if _, err := f.Read(headerStr); err != nil {
		return nil, 0, err
	}

	// Parse shape from header: "('uint8', False, (N, 128))"
	// Find the tuple with two numbers
	hdr := string(headerStr)
	shapeStart := -1
	for i := len(hdr) - 1; i >= 0; i-- {
		if hdr[i] == '(' {
			shapeStart = i
			break
		}
	}
	if shapeStart < 0 {
		return nil, 0, fmt.Errorf("cannot find shape in npy header")
	}
	shapeStr := hdr[shapeStart:]
	// Parse "1056, 128)"
	n1, n2 := 0, 0
	fmt.Sscanf(shapeStr, "(%d, %d)", &n1, &n2)
	if n1 == 0 || n2 != 128 {
		return nil, 0, fmt.Errorf("unexpected shape %d,%d in npy", n1, n2)
	}

	// Read data
	data := make([]byte, n1*128)
	if _, err := f.Read(data); err != nil {
		return nil, 0, err
	}

	return data, n1, nil
}

// loadNormStats reads normalization min/max from a .norm.bin file.
// Format: uint32 dim, float32[dim] min, float32[dim] max
func (pc *PatchClassifier) loadNormStats(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Read dimension
	var dim uint32
	if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
		return fmt.Errorf("read dim: %w", err)
	}
	if dim != Feature784Dim {
		return fmt.Errorf("expected dim %d, got %d", Feature784Dim, dim)
	}

	// Read min array
	for i := 0; i < Feature784Dim; i++ {
		var v float32
		if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
			return fmt.Errorf("read min[%d]: %w", i, err)
		}
		pc.featMin[i] = float64(v)
	}

	// Read max array
	for i := 0; i < Feature784Dim; i++ {
		var v float32
		if err := binary.Read(f, binary.LittleEndian, &v); err != nil {
			return fmt.Errorf("read max[%d]: %w", i, err)
		}
		pc.featMax[i] = float64(v)
	}

	return nil
}

// Predict classifies a 784-dim feature vector and returns the best patch.
func (pc *PatchClassifier) Predict(features [Feature784Dim]float64) PatchPrediction {
	// Normalize features
	inputData := pc.inputTensor.GetData()
	for i := 0; i < Feature784Dim; i++ {
		v := features[i]
		min := pc.featMin[i]
		max := pc.featMax[i]
		rng := max - min
		if rng < 1e-8 {
			rng = 1.0
		}
		inputData[i] = float32((v - min) / rng)
	}

	// Run inference
	if err := pc.session.Run(); err != nil {
		log.Printf("Classifier inference error: %v", err)
		return PatchPrediction{}
	}

	// Read MIDI logits and apply softmax
	midiData := pc.midiTensor.GetData()
	midiProbs := softmax(midiData, MIDIKeyClasses)
	bestMidi := 0
	bestMidiProb := float64(midiProbs[0])
	for i := 1; i < MIDIKeyClasses; i++ {
		if float64(midiProbs[i]) > bestMidiProb {
			bestMidiProb = float64(midiProbs[i])
			bestMidi = i
		}
	}

	return PatchPrediction{
		MidiKey:    bestMidi + pc.noteMin,
		Confidence: bestMidiProb,
	}
}

// PredictTopK returns the top-K patch predictions.
func (pc *PatchClassifier) PredictTopK(features [Feature784Dim]float64, k int) PatchPrediction {
	pred := pc.Predict(features)

	// Get patch logits
	patchData := pc.patchTensor.GetData()
	n := len(patchData)
	if n == 0 {
		return pred
	}

	probs := softmax(patchData, n)

	// Find top-K
	type idxProb struct {
		idx  int
		prob float64
	}
	var topK []idxProb
	for i := 0; i < n; i++ {
		topK = append(topK, idxProb{i, float64(probs[i])})
	}
	// Partial sort for top-K
	if k > n {
		k = n
	}
	for i := 0; i < k; i++ {
		for j := i + 1; j < len(topK); j++ {
			if topK[j].prob > topK[i].prob {
				topK[i], topK[j] = topK[j], topK[i]
			}
		}
	}

	pred.TopK = make([]PatchMatch, k)
	for i := 0; i < k; i++ {
		idx := topK[i].idx
		pred.TopK[i] = PatchMatch{
			Index:      idx,
			Confidence: topK[i].prob,
			PatchBytes: pc.getPatchBytes(idx),
		}
	}

	if len(topK) > 0 {
		pred.PatchBytes = pred.TopK[0].PatchBytes
		pred.Confidence = topK[0].prob
	}

	return pred
}

// getPatchBytes returns the 128-byte patch at the given index from embedded data.
func (pc *PatchClassifier) getPatchBytes(idx int) []byte {
	if pc.patchBytes == nil || idx < 0 || idx >= pc.numPatches {
		return nil
	}
	offset := idx * 128
	return pc.patchBytes[offset : offset+128]
}

// softmax computes softmax over logits.
func softmax(logits []float32, n int) []float32 {
	result := make([]float32, n)
	maxLogit := float32(-math.MaxFloat32)
	for i := 0; i < n; i++ {
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}
	sum := float32(0)
	for i := 0; i < n; i++ {
		result[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		sum += result[i]
	}
	for i := 0; i < n; i++ {
		result[i] /= sum
	}
	return result
}

// Close releases the ONNX session resources.
func (pc *PatchClassifier) Close() {
	if pc.session != nil {
		pc.session.Destroy()
	}
}

// MIDIKeyClasses is the number of MIDI key output classes (3 octaves).
const MIDIKeyClasses = 37
