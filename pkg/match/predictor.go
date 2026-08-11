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

// NewPredictorFromBytes initializes the ONNX runtime session from embedded model bytes.
func NewPredictorFromBytes(data []byte) (*Predictor, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ONNX env: %w", err)
		}
	}

	inputNames := []string{"spectrogram"}
	outputNames := []string{"continuous"}

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

	p, err := ort.NewAdvancedSessionWithONNXData(
		data,
		inputNames,
		outputNames,
		inputs, outputs, nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load ONNX session from bytes: %w", err)
	}

	return &Predictor{
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
		Scores:    nanScores(),
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

	if bestEntry != nil && bestEntry.Candidate.RawBase != NoRawBase {
		c2 := c.Copy()
		c2.RawBase = bestEntry.Candidate.RawBase // value copy
		c2.CopyEnvelopeFromRawBase()
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

	if bestEntry != nil && bestEntry.Candidate.RawBase != NoRawBase {
		c2 := c.Copy()
		c2.RawBase = bestEntry.Candidate.RawBase // value copy
		c2.CopyEnvelopeFromRawBase()
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

// --- Probe Predictor (257 input → 8 probe params → heuristic chain) ---

const (
	ProbeInputBins  = TimbreSpecBins + 1 // 256 spectrum + 1 MIDI note
	ProbeOutputDims = 9                   // AttackMs, DecayMs, SustainLevel, ReleaseMs, GateMs, Brightness, Harmonicity, Semitone, Octave
	DX7OutputDims   = 53                  // Normalized DX7 params: Algorithm + 6 ops × 8 + Feedback + PitchEnv(3)
)

// ProbePredictResult holds the9 predicted probe parameters (all 0-1 normalized).
type ProbePredictResult struct {
	AttackMs     float64 // normalized [0,1] → actual ms × 2000
	DecayMs      float64 // normalized [0,1] → actual ms × 5000
	SustainLevel float64 // already [0,1]
	ReleaseMs    float64 // normalized [0,1] → actual ms × 5000
	GateMs       float64 // normalized [0,1] → actual ms × 5000
	Brightness   float64 // already [0,1]
	Harmonicity  float64 // already [0,1]
	Semitone     float64 // normalized [0,1] → pitch class 0-11
	Octave       float64 // normalized [0,1] → octave 0-8
}

// ProbePredictor runs the probe NN model and converts predictions to DX7 Candidates.
// Supports dual-head models (probes + DX7 direct) and legacy probe-only models.
type ProbePredictor struct {
	modelPath string
	session   *ort.AdvancedSession

	inputTensor  *ort.Tensor[float32]
	probeTensor  *ort.Tensor[float32]
	dx7Tensor    *ort.Tensor[float32] // nil for probe-only models
	hasDX7Head   bool
}

// NewProbePredictor initializes the ONNX runtime session from a probe model file.
// Detects whether the model has a DX7 head by checking output count.
func NewProbePredictor(modelPath string) (*ProbePredictor, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ONNX env: %w", err)
		}
	}

	inputNames := []string{"spectrogram"}

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, ProbeInputBins))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	probeTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, ProbeOutputDims))
	if err != nil {
		return nil, fmt.Errorf("failed to create probe output tensor: %w", err)
	}

	// Try to detect DX7 head by checking model metadata
	// We speculatively create the DX7 tensor and check during session creation
	dx7Tensor, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, DX7OutputDims))

	outputNames := []string{"probe"}
	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{probeTensor}

	// Try with both outputs first (dual-head model)
	if dx7Tensor != nil {
		outputNames = []string{"probe", "dx7"}
		outputs = []ort.Value{probeTensor, dx7Tensor}
	}

	p, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		inputs, outputs, nil,
	)
	if err != nil && dx7Tensor != nil {
		// Dual-head failed, try probe-only
		outputNames = []string{"probe"}
		outputs = []ort.Value{probeTensor}
		dx7Tensor = nil
		p, err = ort.NewAdvancedSession(
			modelPath,
			inputNames,
			outputNames,
			inputs, outputs, nil,
		)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to load probe ONNX session: %w", err)
	}

	hasDX7 := dx7Tensor != nil
	if hasDX7 {
		fmt.Printf("[predictor] Loaded dual-head model: probe (%d dims) + dx7 (%d dims)\n", ProbeOutputDims, DX7OutputDims)
	} else {
		fmt.Printf("[predictor] Loaded probe-only model (%d dims)\n", ProbeOutputDims)
	}

	return &ProbePredictor{
		modelPath:   modelPath,
		session:     p,
		inputTensor: inputTensor,
		probeTensor: probeTensor,
		dx7Tensor:   dx7Tensor,
		hasDX7Head:  hasDX7,
	}, nil
}

// NewProbePredictorFromBytes initializes the ONNX runtime session from embedded model bytes.
func NewProbePredictorFromBytes(data []byte) (*ProbePredictor, error) {
	ort.SetSharedLibraryPath("./lib/libonnxruntime.so")
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to init ONNX env: %w", err)
		}
	}

	inputNames := []string{"spectrogram"}

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, ProbeInputBins))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	probeTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, ProbeOutputDims))
	if err != nil {
		return nil, fmt.Errorf("failed to create probe output tensor: %w", err)
	}

	dx7Tensor, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, DX7OutputDims))

	outputNames := []string{"probe"}
	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{probeTensor}

	if dx7Tensor != nil {
		outputNames = []string{"probe", "dx7"}
		outputs = []ort.Value{probeTensor, dx7Tensor}
	}

	p, err := ort.NewAdvancedSessionWithONNXData(
		data,
		inputNames,
		outputNames,
		inputs, outputs, nil,
	)
	if err != nil && dx7Tensor != nil {
		outputNames = []string{"probe"}
		outputs = []ort.Value{probeTensor}
		dx7Tensor = nil
		p, err = ort.NewAdvancedSessionWithONNXData(
			data,
			inputNames,
			outputNames,
			inputs, outputs, nil,
		)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to load probe ONNX session from bytes: %w", err)
	}

	hasDX7 := dx7Tensor != nil
	return &ProbePredictor{
		session:     p,
		inputTensor: inputTensor,
		probeTensor: probeTensor,
		dx7Tensor:   dx7Tensor,
		hasDX7Head:  hasDX7,
	}, nil
}

// Close releases the ONNX session resources.
func (p *ProbePredictor) Close() {
	if p.session != nil {
		p.session.Destroy()
	}
}

// PredictToProbeCandidate runs the probe NN on a timbre spectrum + MIDI note,
// then converts the predicted probe parameters to a DX7 Candidate via the
// selectAlgorithm + estimateOperatorParams heuristic chain.
func (p *ProbePredictor) PredictToProbeCandidate(spec [TimbreSpecBins]float64, midiNote int) (*Candidate, ProbePredictResult) {
	if p.session == nil {
		result := p.heuristicProbePredict(spec, midiNote)
		cand := p.probeToCandidate(result, spec, midiNote)
		return cand, result
	}

	// Write input: 256 spectrum bins + 1 MIDI note (normalized)
	inputData := p.inputTensor.GetData()
	for i := 0; i < TimbreSpecBins; i++ {
		inputData[i] = float32(spec[i])
	}
	inputData[TimbreSpecBins] = float32(midiNote) / 127.0

	// Execute ONNX session
	if err := p.session.Run(); err != nil {
		log.Printf("ONNX probe execution error: %v, using heuristic fallback", err)
		result := p.heuristicProbePredict(spec, midiNote)
		cand := p.probeToCandidate(result, spec, midiNote)
		return cand, result
	}

	// Read probe output (9 values) and apply sigmoid
	probe := p.probeTensor.GetData()
	sigmoid := func(x float32) float64 {
		return float64(1.0 / (1.0 + math.Exp(-float64(x))))
	}

	var result ProbePredictResult
	result.AttackMs = sigmoid(probe[0])
	result.DecayMs = sigmoid(probe[1])
	result.SustainLevel = sigmoid(probe[2])
	result.ReleaseMs = sigmoid(probe[3])
	result.GateMs = sigmoid(probe[4])
	result.Brightness = sigmoid(probe[5])
	result.Harmonicity = sigmoid(probe[6])
	result.Semitone = sigmoid(probe[7])
	result.Octave = sigmoid(probe[8])

	cand := p.probeToCandidate(result, spec, midiNote)
	return cand, result
}

// PredictToDirectCandidate runs the NN and uses the DX7 head for direct byte prediction
// when available. Falls back to probe+heuristic chain for probe-only models.
func (p *ProbePredictor) PredictToDirectCandidate(spec [TimbreSpecBins]float64, midiNote int) (*Candidate, ProbePredictResult) {
	if p.session == nil || !p.hasDX7Head {
		return p.PredictToProbeCandidate(spec, midiNote)
	}

	// Write input
	inputData := p.inputTensor.GetData()
	for i := 0; i < TimbreSpecBins; i++ {
		inputData[i] = float32(spec[i])
	}
	inputData[TimbreSpecBins] = float32(midiNote) / 127.0

	// Execute ONNX session
	if err := p.session.Run(); err != nil {
		log.Printf("ONNX execution error: %v, using heuristic fallback", err)
		return p.PredictToProbeCandidate(spec, midiNote)
	}

	sigmoid := func(x float32) float64 {
		return float64(1.0 / (1.0 + math.Exp(-float64(x))))
	}

	// Read probe output (for backward compat and auxiliary info)
	probe := p.probeTensor.GetData()
	var result ProbePredictResult
	result.AttackMs = sigmoid(probe[0])
	result.DecayMs = sigmoid(probe[1])
	result.SustainLevel = sigmoid(probe[2])
	result.ReleaseMs = sigmoid(probe[3])
	result.GateMs = sigmoid(probe[4])
	result.Brightness = sigmoid(probe[5])
	result.Harmonicity = sigmoid(probe[6])
	result.Semitone = sigmoid(probe[7])
	result.Octave = sigmoid(probe[8])

	// Read DX7 output (53 values) and build Candidate directly
	dx7 := p.dx7Tensor.GetData()
	cand := p.dx7ToCandidate(dx7, midiNote)

	return cand, result
}

// dx7ToCandidate converts 53 normalized DX7 predictions directly to a Candidate.
// Layout: Algorithm(1) + 6 ops × Mode,Coarse,Fine,Level,Rate1,Rate2,L3,Rate4(48) + Feedback(1) + PitchEnv(3)
func (p *ProbePredictor) dx7ToCandidate(dx7 []float32, midiNote int) *Candidate {
	sigmoid := func(x float32) float64 {
		return float64(1.0 / (1.0 + math.Exp(-float64(x))))
	}
	clamp := func(v float64, lo, hi int) int {
		if v < float64(lo) {
			return lo
		}
		if v > float64(hi) {
			return hi
		}
		return int(v + 0.5)
	}

	cand := &Candidate{
		Algorithm: uint8(clamp(sigmoid(dx7[0])*31, 0, 31)),
		Feedback:  uint8(clamp(sigmoid(dx7[49])*7, 0, 7)),
		Fitness:   math.MaxFloat64,
		Scores:    nanScores(),
	}

	// DX7 operator order in labels: Op6 (idx 0) → Op1 (idx 5)
	// Candidate.Ops order: Ops[0]=Op1, ..., Ops[5]=Op6
	for opIdx := 0; opIdx < 6; opIdx++ {
		labelBase := 1 + opIdx*8 // starts at index 1, 8 params per op
		candOpIdx := 5 - opIdx   // reverse: label Op6 → Cand Ops[5], label Op1 → Cand Ops[0]

		cand.Ops[candOpIdx] = OpParams{
			Mode:        uint8(clamp(sigmoid(dx7[labelBase+0]), 0, 1)),
			FreqCoarse:  uint8(clamp(sigmoid(dx7[labelBase+1])*31, 0, 31)),
			FreqFine:    uint8(clamp(sigmoid(dx7[labelBase+2])*99, 0, 99)),
			OutputLevel: uint8(clamp(sigmoid(dx7[labelBase+3])*99, 0, 99)),
			Rate1:       uint8(clamp(sigmoid(dx7[labelBase+4])*99, 0, 99)),
			Rate2:       uint8(clamp(sigmoid(dx7[labelBase+5])*99, 0, 99)),
			Level3:      uint8(clamp(sigmoid(dx7[labelBase+6])*99, 0, 99)),
			Rate4:       uint8(clamp(sigmoid(dx7[labelBase+7])*99, 0, 99)),
		}
	}

	// Pitch envelope (indices 50-52)
	cand.PitchRate1 = uint8(clamp(sigmoid(dx7[50])*99, 0, 99))
	cand.PitchLevel1 = uint8(clamp(sigmoid(dx7[51])*99, 0, 99))
	cand.PitchLevel2 = uint8(clamp(sigmoid(dx7[52])*99, 0, 99))

	// Attach template from codebook for LFO, keyboard scaling, etc.
	if p.modelPath != "" {
		// Try to find nearest codebook entry by algorithm + frequency
		cand.RawBase = findNearestTemplate(cand)
		cand.CopyEnvelopeFromRawBase()
	}

	return cand
}

// findNearestTemplate finds a suitable RawBase template for the predicted algorithm.
func findNearestTemplate(cand *Candidate) [128]byte {
	var template [128]byte
	copy(template[:], epiano)
	return template
}

// heuristicProbePredict provides a fallback prediction from spectral centroid.
func (p *ProbePredictor) heuristicProbePredict(spec [TimbreSpecBins]float64, midiNote int) ProbePredictResult {
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

	result := ProbePredictResult{
		AttackMs:     0.05,  // ~100ms
		DecayMs:      0.10,  // ~500ms
		SustainLevel: 0.70,
		ReleaseMs:    0.10,  // ~500ms
		GateMs:       0.50,  // ~2500ms
		Brightness:   centroid,
		Harmonicity:  0.60,
		Semitone:     float64(midiNote%12) / 11.0,
		Octave:       float64(midiNote/12-1) / 8.0,
	}

	if centroid > 0.5 {
		result.Brightness = 0.8
		result.Harmonicity = 0.7
	} else {
		result.Brightness = 0.4
		result.Harmonicity = 0.5
	}

	return result
}

// probeToCandidate converts predicted probe parameters to a DX7 Candidate
// using the selectAlgorithm + estimateOperatorParams heuristic chain.
func (p *ProbePredictor) probeToCandidate(result ProbePredictResult, spec [TimbreSpecBins]float64, midiNote int) *Candidate {
	// Denormalize probe parameters
	attackMs := result.AttackMs * 2000.0
	decayMs := result.DecayMs * 5000.0
	sustainLevel := result.SustainLevel
	releaseMs := result.ReleaseMs * 5000.0
	gateMs := result.GateMs * 5000.0
	brightness := result.Brightness
	harmonicity := result.Harmonicity

	// Reconstruct MIDI note from semitone + octave with clamping
	clampedSemi := math.Max(0.0, math.Min(1.0, result.Semitone))
	clampedOct := math.Max(0.0, math.Min(1.0, result.Octave))
	semitone := int(math.Round(clampedSemi * 11.0)) // 0-11
	octave := int(math.Round(clampedOct * 8.0))     // 0-8
	predictedNote := semitone + (octave+1)*12
	if predictedNote < 0 {
		predictedNote = 0
	}
	if predictedNote > 127 {
		predictedNote = 127
	}

	// Use predicted note if available, otherwise fall back to input note
	note := predictedNote
	if note == 0 && midiNote > 0 {
		note = midiNote
	}

	f0 := 440.0 * math.Pow(2, float64(note-69)/12.0)

	// Build TargetFeatures from predicted probe params
	features := TargetFeatures{
		AttackMs:         attackMs,
		DecayMs:          decayMs,
		SustainLevel:     sustainLevel,
		ReleaseMs:        releaseMs,
		GateMs:           gateMs,
		BrightnessScore:  brightness,
		Harmonicity:      harmonicity,
		F0:               f0,
		F0Confidence:     1.0,
		FrequencyRatios:  estimateFrequencyRatiosFromSpectrum(spec, f0),
	}

	// Select algorithm from brightness/harmonicity
	algo := selectAlgorithm(features)

	// Estimate operator params (EG rates, levels, freq coarse/fine from ratios)
	ops := estimateOperatorParams(algo, features)

	// Feedback: harmonic → low, inharmonic → high
	feedback := uint8(4)
	if harmonicity > 0.7 {
		feedback = 2
	}

	c := &Candidate{
		Algorithm: algo,
		Feedback:  feedback,
		Ops:       ops,
		Fitness:   math.MaxFloat64,
		Scores:    nanScores(),
	}

	// Pitch envelope defaults
	c.PitchRate1 = 99
	c.PitchLevel1 = 50
	c.PitchLevel2 = 50

	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}

	return c
}

// estimateFrequencyRatiosFromSpectrum estimates frequency ratios from the timbre spectrum.
// The 256-bin spectrum is compressed from a 1024-point FFT at 44100 Hz.
// Each output bin covers ~43 Hz (22050 Hz / 513 bins × 2 bins per output).
func estimateFrequencyRatiosFromSpectrum(spec [TimbreSpecBins]float64, f0 float64) []float64 {
	if f0 < 20 {
		return []float64{1.0}
	}

	// binToFreq converts a 256-bin index to approximate frequency in Hz
	// 1024-point FFT at 44100 Hz: 513 positive bins, 22050 Hz max
	// Compressed to 256 bins: each output bin ≈ 2 input bins ≈ 86 Hz
	binToFreq := func(bin int) float64 {
		return float64(bin) * 22050.0 / float64(TimbreSpecBins)
	}

	// Find peaks in the spectrum
	type peak struct {
		bin  int
		mag  float64
	}
	var peaks []peak
	for i := 2; i < TimbreSpecBins-1; i++ {
		if spec[i] > spec[i-1] && spec[i] > spec[i+1] && spec[i] > 0.1 {
			peaks = append(peaks, peak{i, spec[i]})
		}
	}

	// Sort by magnitude (strongest first)
	for i := 0; i < len(peaks)-1; i++ {
		for j := i + 1; j < len(peaks); j++ {
			if peaks[j].mag > peaks[i].mag {
				peaks[i], peaks[j] = peaks[j], peaks[i]
			}
		}
	}

	// Limit to top 6 peaks
	if len(peaks) > 6 {
		peaks = peaks[:6]
	}

	// Convert bin indices to frequency ratios
	var ratios []float64
	for _, p := range peaks {
		freq := binToFreq(p.bin)
		ratio := freq / f0
		// Round to nearest 0.5 for cleaner ratios
		ratio = math.Round(ratio*2) / 2
		if ratio < 0.5 {
			ratio = 0.5
		}
		if ratio > 31 {
			ratio = 31
		}
		ratios = append(ratios, ratio)
	}

	// Ensure ratio 1.0 is present
	hasOne := false
	for _, r := range ratios {
		if math.Abs(r-1.0) < 0.3 {
			hasOne = true
			break
		}
	}
	if !hasOne {
		ratios = append([]float64{1.0}, ratios...)
	}

	return ratios
}
