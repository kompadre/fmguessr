package match

import (
	"fmt"
	"log"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// TimbreLabels is the number of timbre parameters the CNN predicts.
// Layout: [Algorithm(1), FreqCoarse(6), FreqFine(6), OutputLevel(6), Mode(6), Feedback(1)] = 20
// The remaining 30 positions are envelope params set to neutral (0 or 1.0).
const TimbreLabels = 50 // Keep 50 for compatibility, envelope params neutralized

type Predictor struct {
	modelPath string
	session   *ort.AdvancedSession

	// ONNX tensors
	inputTensor      *ort.Tensor[float32]
	continuousTensor *ort.Tensor[float32]
}

// PredictResult holds the raw neural network output for a single timbre spectrum.
type PredictResult struct {
	// Continuous params predicted by NN
	FreqFine    [6]float64 // 0-1 normalized
	OutputLevel [6]float64 // 0-1 normalized
	Feedback    float64    // 0-1 normalized
	Confidence  float64
}

// NewPredictor initializes the ONNX runtime session from a timbre model.
func NewPredictor(modelPath string) (*Predictor, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ONNX env: %w", err)
		}
	}

	inputNames := []string{"spectrogram"}
	outputNames := []string{"continuous"}

	// Input: (1, 1, TimbreSpecBins) for 1D timbre spectrum with channel dim
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, TimbreSpecBins))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	continuousTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 13))
	if err != nil {
		return nil, fmt.Errorf("failed to create continuous tensor: %w", err)
	}

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{continuousTensor}

	p, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		inputs, outputs, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX session: %w", err)
	}

	return &Predictor{
		modelPath:        modelPath,
		session:          p,
		inputTensor:      inputTensor,
		continuousTensor: continuousTensor,
	}, nil
}

// Close releases the ONNX session resources.
func (p *Predictor) Close() {
	if p.session != nil {
		p.session.Destroy()
	}
}

// PredictToCandidate runs the NN on a timbre spectrum and returns a Candidate.
// The NN predicts continuous params (fines, levels, feedback).
// Algorithm and coarse ratios are set to defaults (to be optimized by GA).
func (p *Predictor) PredictToCandidate(spec [TimbreSpecBins]float64) (*Candidate, PredictResult) {
	if p.session == nil {
		result := p.heuristicPredict(spec)
		cand := p.resultToCandidate(result)
		return cand, result
	}

	// Write timbre spectrum into input tensor (1, 1, 256)
	inputData := p.inputTensor.GetData()
	for i := 0; i < TimbreSpecBins; i++ {
		inputData[i] = float32(spec[i])
	}

	// Execute ONNX session
	if err := p.session.Run(); err != nil {
		log.Printf("ONNX execution error: %v, using heuristic fallback", err)
		result := p.heuristicPredict(spec)
		cand := p.resultToCandidate(result)
		return cand, result
	}

	// Read continuous output (13 values: 6 fines + 6 levels + 1 feedback)
	continuous := p.continuousTensor.GetData()

	var result PredictResult
	result.Confidence = 0.95

	// Model outputs raw logits — apply sigmoid to get [0,1] range.
	sigmoid := func(x float32) float64 {
		return float64(1.0 / (1.0 + math.Exp(-float64(x))))
	}

	// Extract fines (indices 0-5)
	for op := 0; op < 6; op++ {
		result.FreqFine[op] = sigmoid(continuous[op])
	}

	// Extract levels (indices 6-11)
	for op := 0; op < 6; op++ {
		result.OutputLevel[op] = sigmoid(continuous[6+op])
	}

	// Extract feedback (index 12)
	result.Feedback = sigmoid(continuous[12])

	cand := p.resultToCandidate(result)
	return cand, result
}

// heuristicPredict provides a fallback prediction from a timbre spectrum.
func (p *Predictor) heuristicPredict(spec [TimbreSpecBins]float64) PredictResult {
	result := PredictResult{
		Confidence: 0.5,
	}

	// Analyze spectral centroid from timbre spectrum
	totalEnergy := 0.0
	weightedSum := 0.0
	for i, v := range spec {
		totalEnergy += v
		weightedSum += v * float64(i)
	}
	centroid := 0.0
	if totalEnergy > 1e-10 {
		centroid = weightedSum / totalEnergy / float64(len(spec))
	}

	// Default continuous params
	for op := 0; op < 6; op++ {
		result.FreqFine[op] = 0.0
		result.OutputLevel[op] = 0.85
	}

	// Bright sounds use higher output levels
	if centroid > 0.5 {
		for op := 0; op < 6; op++ {
			result.OutputLevel[op] = 0.90
		}
		result.Feedback = 5.0 / 7.0
	} else {
		result.Feedback = 3.0 / 7.0
	}

	return result
}

// resultToCandidate converts a PredictResult to a Candidate.
// Sets coarse ratios to defaults (algorithm 15, coarse=1).
func (p *Predictor) resultToCandidate(result PredictResult) *Candidate {
	c := &Candidate{
		Algorithm: 14, // Default Algorithm 15 (0-indexed)
		Feedback:  uint8(math.Round(result.Feedback * 7.0)),
		Fitness:   math.MaxFloat64,
	}

	for op := 0; op < 6; op++ {
		c.Ops[op] = OpParams{
			Mode:        0,
			FreqCoarse:  1, // Default coarse
			FreqFine:    uint8(math.Round(result.FreqFine[op] * 99.0)),
			OutputLevel: uint8(math.Round(result.OutputLevel[op] * 99.0)),
			Rate1:       0,  // Neutral envelope
			Rate2:       0,
			Level3:      99,
			Rate4:       0,
		}
	}

	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}

	return c
}

// AttachNearestRawBase finds the closest codebook entry and copies its RawBase to the candidate.
func AttachNearestRawBase(c *Candidate, codebook *Codebook) *Candidate {
	if codebook == nil || len(codebook.Entries) == 0 {
		return c
	}

	bestDist := 999999.0
	var bestEntry *CodebookEntry

	for i := range codebook.Entries {
		entry := &codebook.Entries[i]
		dist := 0.0

		if entry.Candidate.Algorithm != c.Algorithm {
			dist += 100.0
		}

		for op := 0; op < 6; op++ {
			d := float64(entry.Candidate.Ops[op].FreqCoarse) - float64(c.Ops[op].FreqCoarse)
			dist += d * d
		}

		if dist < bestDist {
			bestDist = dist
			bestEntry = entry
		}
	}

	if bestEntry != nil && len(bestEntry.Candidate.RawBase) == 128 {
		c2 := c.Copy()
		c2.RawBase = make([]byte, 128)
		copy(c2.RawBase, bestEntry.Candidate.RawBase)
		return c2
	}

	return c
}

// AttachNearestRawBaseTimbre finds the closest timbre codebook entry and copies its RawBase.
func AttachNearestRawBaseTimbre(c *Candidate, tc *TimbreCodebook) *Candidate {
	if tc == nil || len(tc.Entries) == 0 {
		return c
	}

	bestDist := 999999.0
	var bestEntry *TimbreCodebookEntry

	for i := range tc.Entries {
		entry := &tc.Entries[i]
		dist := 0.0

		if entry.Candidate.Algorithm != c.Algorithm {
			dist += 100.0
		}

		for op := 0; op < 6; op++ {
			d := float64(entry.Candidate.Ops[op].FreqCoarse) - float64(c.Ops[op].FreqCoarse)
			dist += d * d
		}

		if dist < bestDist {
			bestDist = dist
			bestEntry = &tc.Entries[i]
		}
	}

	if bestEntry != nil && len(bestEntry.Candidate.RawBase) == 128 {
		c2 := c.Copy()
		c2.RawBase = make([]byte, 128)
		copy(c2.RawBase, bestEntry.Candidate.RawBase)
		return c2
	}

	return c
}

func argmax(slice []float32) int {
	bestIdx := 0
	bestVal := slice[0]
	for i := 1; i < len(slice); i++ {
		if slice[i] > bestVal {
			bestVal = slice[i]
			bestIdx = i
		}
	}
	return bestIdx
}
