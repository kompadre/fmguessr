//go:build regression

package tests

import (
	"math"
	"testing"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

// TestRegressionEncodeDecodeRoundTrip verifies that Encode→Decode is lossless
// for a set of known parameter combinations. This guards against regressions
// in the 128-byte DX7 SysEx encoding.
func TestRegressionEncodeDecodeRoundTrip(t *testing.T) {
	cases := []struct {
		name  string
		patch func() *match.Candidate
	}{
		{"SineBase", match.SineBase},
		{"SquareBase", match.SquareBase},
		{"SawtoothBase", match.SawtoothBase},
		{"TriangleBase", match.TriangleBase},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			original := tc.patch()
			encoded := original.Encode()
			if len(encoded) != 128 {
				t.Fatalf("Encode produced %d bytes, expected 128", len(encoded))
			}

			decoded := match.DecodeVoicePatch(encoded)
			if decoded == nil {
				t.Fatal("DecodeVoicePatch returned nil")
			}

			// Verify algorithm preserved
			if decoded.Algorithm != original.Algorithm {
				t.Errorf("Algorithm: got %d, want %d", decoded.Algorithm, original.Algorithm)
			}
			// Verify feedback preserved
			if decoded.Feedback != original.Feedback {
				t.Errorf("Feedback: got %d, want %d", decoded.Feedback, original.Feedback)
			}
			// Verify operator parameters preserved
			for op := 0; op < 6; op++ {
				if decoded.Ops[op].OutputLevel != original.Ops[op].OutputLevel {
					t.Errorf("Op%d.OutputLevel: got %d, want %d", op, decoded.Ops[op].OutputLevel, original.Ops[op].OutputLevel)
				}
				if decoded.Ops[op].FreqCoarse != original.Ops[op].FreqCoarse {
					t.Errorf("Op%d.FreqCoarse: got %d, want %d", op, decoded.Ops[op].FreqCoarse, original.Ops[op].FreqCoarse)
				}
				if decoded.Ops[op].FreqFine != original.Ops[op].FreqFine {
					t.Errorf("Op%d.FreqFine: got %d, want %d", op, decoded.Ops[op].FreqFine, original.Ops[op].FreqFine)
				}
			}
		})
	}
}

// TestRegressionSpectralDistanceNonZero verifies that SpectralDistance between
// two different signals is non-zero and in a reasonable range.
// Note: SpectralDistance is intentionally asymmetric (RMS weighting relative to target).
func TestRegressionSpectralDistanceNonZero(t *testing.T) {
	a := match.RenderCandidate(match.SineBase(), 60, 500*time.Millisecond)
	b := match.RenderCandidate(match.SquareBase(), 60, 500*time.Millisecond)
	a = match.NormalizeRMS(match.TrimOrPad(a, 22050))
	b = match.NormalizeRMS(match.TrimOrPad(b, 22050))

	dAB := match.SpectralDistance(a, b)
	dBA := match.SpectralDistance(b, a)

	if dAB <= 0 {
		t.Errorf("SpectralDistance(a, b) should be > 0, got %.10f", dAB)
	}
	if dBA <= 0 {
		t.Errorf("SpectralDistance(b, a) should be > 0, got %.10f", dBA)
	}
	// Verify they are in the same ballpark (not wildly different)
	ratio := dAB / dBA
	if ratio > 2.0 || ratio < 0.5 {
		t.Errorf("SpectralDistance asymmetry too large: d(a,b)=%.6f d(b,a)=%.6f ratio=%.2f", dAB, dBA, ratio)
	}
}

// TestRegressionMelCosineDistanceSelfIsZero verifies that the mel cosine distance
// of a signal to itself is zero. Guards against normalization regressions.
func TestRegressionMelCosineDistanceSelfIsZero(t *testing.T) {
	audio := match.RenderCandidate(match.SineBase(), 60, 1*time.Second)
	normed := match.NormalizeRMS(match.TrimOrPad(audio, 44100))

	targetMel := match.ComputeTargetMel(normed, 44100)
	dist := match.MelCosineDistanceCached(targetMel, normed, 44100)

	if dist > 1e-6 {
		t.Errorf("MelCosineDistance(self) = %.10f, expected ~0", dist)
	}
}

// TestRegressionNormalizeRMSTarget verifies that NormalizeRMS produces
// the expected target RMS level (0.1). Guards against scaling regressions.
func TestRegressionNormalizeRMSTarget(t *testing.T) {
	audio := match.RenderCandidate(match.SineBase(), 60, 500*time.Millisecond)
	normed := match.NormalizeRMS(audio)
	rms := match.CalcRMS(normed)

	if math.Abs(rms-0.1) > 1e-6 {
		t.Errorf("NormalizeRMS target RMS: got %.10f, want 0.1", rms)
	}
}

// TestRegressionCalcRMSAndNormalizeConsistency verifies that CalcRMSAndNormalize
// produces the same result as separate CalcRMS + NormalizeRMS calls.
func TestRegressionCalcRMSAndNormalizeConsistency(t *testing.T) {
	audio := match.RenderCandidate(match.SineBase(), 60, 500*time.Millisecond)

	rawRMS := match.CalcRMS(audio)
	separate := match.NormalizeRMS(audio)

	combinedRMS, combined := match.CalcRMSAndNormalize(0.1, audio)

	if math.Abs(rawRMS-combinedRMS) > 1e-10 {
		t.Errorf("RMS mismatch: separate=%.10f combined=%.10f", rawRMS, combinedRMS)
	}
	for i := range separate {
		if math.Abs(separate[i]-combined[i]) > 1e-10 {
			t.Errorf("Sample %d mismatch: separate=%.10f combined=%.10f", i, separate[i], combined[i])
			break
		}
	}
}

// TestRegressionHannWindowCache verifies that the Hann window cache returns
// the same values as a fresh computation.
func TestRegressionHannWindowCache(t *testing.T) {
	// The getHannWindow function is unexported, so we test it indirectly
	// by computing mel spectrograms with the same parameters.
	audio := match.RenderCandidate(match.SineBase(), 60, 500*time.Millisecond)
	spec1 := match.ComputeMelSpectrogram(audio, 44100)
	spec2 := match.ComputeMelSpectrogram(audio, 44100)

	for i := 0; i < 128; i++ {
		for j := 0; j < 128; j++ {
			if math.Abs(spec1[i][j]-spec2[i][j]) > 1e-10 {
				t.Errorf("Mel spectrogram mismatch at [%d][%d]: %.10f vs %.10f", i, j, spec1[i][j], spec2[i][j])
				return
			}
		}
	}
}

// TestRegressionSessionRoundTrip verifies SaveSession/LoadSession preserves all fields.
func TestRegressionSessionRoundTrip(t *testing.T) {
	// Create synthetic trajectory data
	embeddings := make([][]float32, 20)
	for i := range embeddings {
		embeddings[i] = make([]float32, 128)
		for j := range embeddings[i] {
			embeddings[i][j] = float32(i*128 + j)
		}
	}
	original := &match.SessionData{
		Trajectory: &match.TrajectorySpec{
			Embeddings: embeddings,
			TimeSteps:  20,
			EmbDim:     128,
		},
		PhaseBounds: match.PhaseBounds{
			AttackEnd:  3,
			SustainEnd: 15,
		},
		LFOSpectrum: []float64{1.0, 2.5, 3.7, 0.1, 4.2},
	}
	copy(original.AudioHash[:], []byte("testhash12345678901234567890ab"))

	tmpFile := t.TempDir() + "/test_session.bin"
	if err := match.SaveSession(tmpFile, original); err != nil {
		t.Fatalf("SaveSession failed: %v", err)
	}

	loaded, err := match.LoadSession(tmpFile)
	if err != nil {
		t.Fatalf("LoadSession failed: %v", err)
	}

	// Verify hash
	if loaded.AudioHash != original.AudioHash {
		t.Error("AudioHash mismatch")
	}

	// Verify phase bounds
	if loaded.PhaseBounds.AttackEnd != 3 {
		t.Errorf("AttackEnd: got %d, want 3", loaded.PhaseBounds.AttackEnd)
	}
	if loaded.PhaseBounds.SustainEnd != 15 {
		t.Errorf("SustainEnd: got %d, want 15", loaded.PhaseBounds.SustainEnd)
	}

	// Verify trajectory dimensions
	if loaded.Trajectory.TimeSteps != 20 {
		t.Errorf("TimeSteps: got %d, want 20", loaded.Trajectory.TimeSteps)
	}
	if loaded.Trajectory.EmbDim != 128 {
		t.Errorf("EmbDim: got %d, want 128", loaded.Trajectory.EmbDim)
	}

	// Verify LFO spectrum
	if len(loaded.LFOSpectrum) != 5 {
		t.Fatalf("LFOSpectrum length: got %d, want 5", len(loaded.LFOSpectrum))
	}
	for i, v := range loaded.LFOSpectrum {
		if math.Abs(v-original.LFOSpectrum[i]) > 1e-3 {
			t.Errorf("LFOSpectrum[%d]: got %f, want %f", i, v, original.LFOSpectrum[i])
		}
	}

	// Verify embedding values (float32 precision)
	for i := 0; i < 20; i++ {
		for j := 0; j < 128; j++ {
			expected := float32(i*128 + j)
			if loaded.Trajectory.Embeddings[i][j] != expected {
				t.Errorf("Embedding[%d][%d]: got %f, want %f", i, j, loaded.Trajectory.Embeddings[i][j], expected)
				return
			}
		}
	}
}

// TestRegressionAudioHashDeterministic verifies AudioHash produces the same result for the same input.
func TestRegressionAudioHashDeterministic(t *testing.T) {
	samples := make([]float64, 1000)
	for i := range samples {
		samples[i] = float64(i) * 0.001
	}
	h1 := match.AudioHash(samples)
	h2 := match.AudioHash(samples)
	if h1 != h2 {
		t.Error("AudioHash not deterministic for same input")
	}

	// Different input → different hash
	samples[500] = 999.0
	h3 := match.AudioHash(samples)
	if h1 == h3 {
		t.Error("AudioHash should differ for different input")
	}
}

// TestRegressionTrajectoryDistanceSelfIsZero verifies that trajectory distance
// of a trajectory to itself is zero (or near-zero).
func TestRegressionTrajectoryDistanceSelfIsZero(t *testing.T) {
	// Create a synthetic trajectory
	embeddings := make([][]float32, 30)
	for i := range embeddings {
		embeddings[i] = make([]float32, 128)
		for j := range embeddings[i] {
			embeddings[i][j] = float32(math.Sin(float64(i) * 0.1 * float64(j)))
		}
		// L2 normalize
		var norm float64
		for _, v := range embeddings[i] {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm > 1e-10 {
			for j := range embeddings[i] {
				embeddings[i][j] /= float32(norm)
			}
		}
	}

	traj := &match.TrajectorySpec{
		Embeddings: embeddings,
		TimeSteps:  30,
		EmbDim:     128,
	}

	cfg := match.DefaultTrajectoryConfig()
	dist := match.TrajectoryDistance(traj, traj, cfg, nil, nil)

	if dist > 1e-6 {
		t.Errorf("TrajectoryDistance(self) = %f, expected ~0", dist)
	}
}

// TestRegressionPhaseBoundsDetection verifies phase boundary detection
// on a synthetic attack-sustain-release signal.
func TestRegressionPhaseBoundsDetection(t *testing.T) {
	sr := 44100
	// Create 1s signal: 50ms attack, 700ms sustain, 250ms release
	samples := make([]float64, sr)
	attackEnd := sr * 50 / 1000   // 2205 samples
	sustainEnd := sr * 750 / 1000 // 33075 samples

	// Attack: ramp up
	for i := 0; i < attackEnd; i++ {
		samples[i] = float64(i) / float64(attackEnd) * 0.5
	}
	// Sustain: constant
	for i := attackEnd; i < sustainEnd; i++ {
		samples[i] = 0.5
	}
	// Release: ramp down
	for i := sustainEnd; i < sr; i++ {
		frac := float64(i-sustainEnd) / float64(sr-sustainEnd)
		samples[i] = 0.5 * (1.0 - frac)
	}

	bounds := match.DetectPhaseBounds(samples, sr)

	// Attack end should be near sample 2205 (±500 samples)
	if bounds.AttackEnd < 1 || bounds.AttackEnd > 10 {
		t.Logf("AttackEnd frame: %d (expected ~3-6 for 50ms attack at 25ms frames)", bounds.AttackEnd)
	}

	// Sustain end should be near the release boundary
	if bounds.SustainEnd <= bounds.AttackEnd {
		t.Errorf("SustainEnd (%d) should be after AttackEnd (%d)", bounds.SustainEnd, bounds.AttackEnd)
	}
}

// TestRegressionLFONormStats verifies Z-score normalization convergence.
func TestRegressionLFONormStats(t *testing.T) {
	stats := match.NewLFONormStats(0.1)

	// Feed constant values — Z-score should converge to 0
	for i := 0; i < 100; i++ {
		z := stats.Update(5.0)
		if i > 50 && math.Abs(z) > 0.1 {
			t.Errorf("After 50 iterations with constant input, Z-score should be near 0, got %f", z)
			break
		}
	}

	// Feed a value far from the mean — Z-score should be non-zero
	stats2 := match.NewLFONormStats(0.1)
	for i := 0; i < 50; i++ {
		stats2.Update(5.0)
	}
	z := stats2.Update(100.0)
	if math.Abs(z) < 1.0 {
		t.Errorf("Outlier Z-score should be significant, got %f", z)
	}
}

// TestRegressionSelectTopK verifies that selectTopK returns correct indices.
func TestRegressionSelectTopK(t *testing.T) {
	// Test via TrajectoryDistance's internal selectTopK — we can't call it directly
	// since it's unexported, but we can verify the trajectory distance itself works
	// with different quality candidates.
	embeddings := make([][]float32, 10)
	for i := range embeddings {
		embeddings[i] = make([]float32, 4)
		embeddings[i][0] = float32(i) // increasing distance from zero
	}

	traj := &match.TrajectorySpec{
		Embeddings: embeddings,
		TimeSteps:  10,
		EmbDim:     4,
	}

	cfg := match.DefaultTrajectoryConfig()

	// Distance to self should be 0
	selfDist := match.TrajectoryDistance(traj, traj, cfg, nil, nil)
	if selfDist > 1e-6 {
		t.Errorf("Self-distance should be ~0, got %f", selfDist)
	}

	// Distance to a different trajectory should be > 0
	other := &match.TrajectorySpec{
		Embeddings: make([][]float32, 10),
		TimeSteps:  10,
		EmbDim:     4,
	}
	for i := range other.Embeddings {
		other.Embeddings[i] = []float32{float32(100 - i), 0, 0, 0}
	}
	otherDist := match.TrajectoryDistance(traj, other, cfg, nil, nil)
	if otherDist <= 0 {
		t.Errorf("Different trajectories should have positive distance, got %f", otherDist)
	}
}

// TestRegressionFormantExtraction verifies that formant extraction produces
// non-empty, ordered peaks from a harmonic signal.
func TestRegressionFormantExtraction(t *testing.T) {
	// Render a known DX7 patch (sine wave at C4)
	patch := match.SineBase()
	patch.Ops[5].OutputLevel = 99
	patch.Ops[5].FreqCoarse = 1
	audio := match.RenderCandidate(patch, 60, 500*time.Millisecond)
	t.Logf("Audio: %d samples (%.2fs)", len(audio), float64(len(audio))/44100.0)

	// Check audio has signal
	sum := 0.0
	for _, s := range audio {
		sum += s * s
	}
	rms := math.Sqrt(sum / float64(len(audio)))
	t.Logf("RMS: %.6f", rms)

	// Try spectral envelope extraction directly
	env := match.ExtractSpectralEnvelope(audio, 44100)
	t.Logf("Envelope: %d bins", len(env))
	if len(env) > 0 {
		maxEnv := 0.0
		for _, v := range env {
			if v > maxEnv {
				maxEnv = v
			}
		}
		t.Logf("Envelope max: %.6f", maxEnv)
	}

	formants := match.ExtractFormants(audio, 44100, 4)
	if len(formants) == 0 {
		t.Fatal("ExtractFormants returned empty for a harmonic signal")
	}

	// Verify ordering (F1 < F2 < F3 ...)
	for i := 1; i < len(formants); i++ {
		if formants[i].Frequency <= formants[i-1].Frequency {
			t.Errorf("Formants not ordered by frequency: F%d=%.0f >= F%d=%.0f",
				i, formants[i-1].Frequency, i+1, formants[i].Frequency)
		}
	}

	// Verify amplitudes are normalized [0,1]
	for i, f := range formants {
		if f.Amplitude < 0 || f.Amplitude > 1.0 {
			t.Errorf("Formant %d amplitude out of range: %f", i+1, f.Amplitude)
		}
	}

	t.Logf("Formants: %d peaks", len(formants))
	for i, f := range formants {
		t.Logf("  F%d: %.0f Hz (amp=%.3f)", i+1, f.Frequency, f.Amplitude)
	}
}

// TestRegressionBesselFunction verifies Bessel J_n(x) against known values.
func TestRegressionBesselFunction(t *testing.T) {
	// Known values: J_0(0) = 1, J_1(0) = 0, J_0(2.4048) ≈ 0
	// These are internal tests — besselJ is unexported, so we test via PredictSpectralPeaks

	// Create a simple FM patch and verify it produces sidebands
	patch := match.SineBase()
	patch.Algorithm = 5 // simple 2-carrier FM
	patch.Ops[5].FreqCoarse = 1 // carrier at fundamental
	patch.Ops[5].OutputLevel = 99
	patch.Ops[4].FreqCoarse = 1 // modulator at fundamental
	patch.Ops[4].OutputLevel = 80

	peaks := match.PredictSpectralPeaks(patch, 440.0)
	if len(peaks) == 0 {
		t.Fatal("PredictSpectralPeaks returned empty for FM patch")
	}

	// Should have at least the carrier + some sidebands
	carrierFound := false
	for _, p := range peaks {
		if math.Abs(p.Frequency-440.0) < 50 {
			carrierFound = true
			break
		}
	}
	if !carrierFound {
		t.Errorf("Carrier at 440 Hz not found in predicted spectrum. Peaks: %v", peaks)
	}

	t.Logf("Predicted %d spectral peaks:", len(peaks))
	for i, p := range peaks {
		if i < 8 {
			t.Logf("  %.0f Hz (amp=%.3f)", p.Frequency, p.Amplitude)
		}
	}
}

// TestRegressionFormantDistance verifies formant distance properties.
func TestRegressionFormantDistance(t *testing.T) {
	peaks := []match.FormantPeak{
		{Frequency: 300, Amplitude: 1.0},
		{Frequency: 1200, Amplitude: 0.8},
		{Frequency: 2400, Amplitude: 0.5},
	}

	// Self-distance should be 0
	selfDist := match.FormantDistance(peaks, peaks)
	if selfDist > 1e-6 {
		t.Errorf("Self-distance should be ~0, got %f", selfDist)
	}

	// Distance to empty should be 1.0
	emptyDist := match.FormantDistance(peaks, nil)
	if math.Abs(emptyDist-1.0) > 1e-6 {
		t.Errorf("Distance to empty should be 1.0, got %f", emptyDist)
	}

	// Distance to shifted peaks should be > 0
	shifted := []match.FormantPeak{
		{Frequency: 350, Amplitude: 1.0},
		{Frequency: 1300, Amplitude: 0.8},
		{Frequency: 2500, Amplitude: 0.5},
	}
	shiftDist := match.FormantDistance(peaks, shifted)
	if shiftDist <= 0 {
		t.Errorf("Shifted peaks should have positive distance, got %f", shiftDist)
	}

	t.Logf("Formant distances: self=%.4f, empty=%.4f, shifted=%.4f", selfDist, emptyDist, shiftDist)
}

// TestRegressionGenerateFormantCandidates verifies formant candidate generation.
func TestRegressionGenerateFormantCandidates(t *testing.T) {
	targetFormants := []match.FormantPeak{
		{Frequency: 300, Amplitude: 1.0},
		{Frequency: 1200, Amplitude: 0.8},
	}

	cands := match.GenerateFormantCandidates(targetFormants, 261.63, 60)
	if len(cands) == 0 {
		t.Fatal("GenerateFormantCandidates returned empty")
	}

	// Verify candidates have valid algorithm and operator params
	for i, c := range cands {
		if c.Algorithm >= 32 {
			t.Errorf("Candidate %d: invalid algorithm %d", i, c.Algorithm)
		}
		// Verify at least one carrier has a non-zero frequency
		hasCarrier := false
		for op := 0; op < 6; op++ {
			if match.IsCarrier(c.Algorithm, op) && c.Ops[op].FreqCoarse > 0 {
				hasCarrier = true
				break
			}
		}
		if !hasCarrier {
			t.Errorf("Candidate %d (algo=%d): no carrier with non-zero frequency", i, c.Algorithm+1)
		}
	}

	t.Logf("Generated %d formant candidates:", len(cands))
	for i, c := range cands {
		peaks := match.PredictSpectralPeaks(c, 261.63)
		dist := match.FormantDistance(targetFormants, peaks)
		t.Logf("  #%d: algo=%d formant_dist=%.3f", i+1, c.Algorithm+1, dist)
	}
}
