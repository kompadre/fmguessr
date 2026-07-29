package tests

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kompadre/fmguessr/pkg/match"
)

func TestEndToEndPipeline(t *testing.T) {
	// 0. Build binaries
	t.Log("Building binaries...")
	buildCmd := exec.Command("go", "build", "-o", "../match", "../cmd/match/main.go")
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to build match: %v\n%s", err, string(out))
	}
	buildCmd = exec.Command("go", "build", "-o", "../index", "../cmd/index/main.go")
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to build index: %v\n%s", err, string(out))
	}

	workDir, _ := os.Getwd()
	parentDir := filepath.Dir(workDir)
	binMatch := filepath.Join(parentDir, "match")
	binIndex := filepath.Join(parentDir, "index")
	syxFile := filepath.Join(parentDir, "Dexed_01.syx")
	targetWav := filepath.Join(parentDir, "bla3.wav")
	codebookFile := filepath.Join(parentDir, "test_codebook.bin")
	resultSyx := filepath.Join(parentDir, "test_out.syx")
	resultWav := filepath.Join(parentDir, "test_match.wav")

	defer os.Remove(codebookFile)
	defer os.Remove(resultSyx)
	defer os.Remove(resultWav)

	// 1. Test Codebook Indexer
	t.Run("Indexer", func(t *testing.T) {
		t.Log("Testing indexer creation...")
		cmd := exec.Command(binIndex, "-overwrite", "-o", codebookFile, syxFile)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("Index failed: %v\n%s", err, string(out))
		}

		// Validate File
		info, err := os.Stat(codebookFile)
		if err != nil {
			t.Fatalf("Codebook not created: %v", err)
		}
		// 12 byte header + 32 * (16+128+256) byte entries = 12812
		if info.Size() != 12812 {
			t.Errorf("Unexpected codebook size: %d, expected 12812", info.Size())
		}

		// Verify Magic
		f, _ := os.Open(codebookFile)
		magic := make([]byte, 4)
		f.Read(magic)
		f.Close()
		if string(magic) != "DXCB" {
			t.Errorf("Invalid magic: %s", string(magic))
		}

		// Test Deduplication
		t.Log("Testing deduplication...")
		cmd = exec.Command(binIndex, "-o", codebookFile, syxFile)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("Append failed: %v\n%s", err, string(out))
		}
		if !strings.Contains(string(out), "32 duplicates skipped") {
			t.Errorf("Deduplication failed, output: %s", string(out))
		}
	})

	// 2. Test Matcher
	t.Run("Matcher", func(t *testing.T) {
		t.Log("Testing matcher...")
		// Use few generations for speed since LAURIE should be found in Gen 0 via KNN
		cmd := exec.Command(binMatch, "-codebook", codebookFile, "-generations", "5", "-syx", resultSyx, "-v", targetWav)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("Match failed: %v\n%s", err, string(out))
		}

		if !strings.Contains(string(out), "Auto-detected root note: C3") {
			t.Logf("Warning: Auto-detect note log not found, might be confidence issue. Output: %s", string(out))
		}

		// Check output sysex
		info, err := os.Stat(resultSyx)
		if err != nil || info.Size() != 4104 {
			t.Fatalf("Invalid result sysex: %v", err)
		}

		// Verify LAURIE recovery (Slot 1 in Dexed_01 is Alg 15)
		// We expect the GA to keep Alg 15 (index 14) as the best
		if !strings.Contains(string(out), "algo=14") {
			// Note: result.Best.String() prints index 0-31, so Algo 15 is "algo=14"
			t.Errorf("Failed to recover correct algorithm. Output: %s", string(out))
		}
	})

	// 3. Test Playback & Audio Integrity
	t.Run("Playback", func(t *testing.T) {
		t.Log("Testing playback render...")
		// go run cmd/play/main.go -o test_match.wav test_out.syx 0 C3
		cmd := exec.Command("go", "run", "../cmd/play/main.go", "-o", resultWav, resultSyx, "0", "C3")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("Play failed: %v\n%s", err, string(out))
		}

		samples, _, err := match.LoadWav(resultWav)
		if err != nil {
			t.Fatalf("Could not load result wav: %v", err)
		}

		// Check duration (approx 3s: 2s sustain + tail)
		duration := float64(len(samples)) / 44100.0
		if duration < 2.0 {
			t.Errorf("Audio too short: %.2fs", duration)
		}

		// Check Sustain Energy (RMS in 100ms blocks)
		frameSize := 4410
		for i := 0; i < 15; i++ { // check first 1.5 seconds
			start := i * frameSize
			end := start + frameSize
			sum := 0.0
			for j := start; j < end; j++ {
				sum += samples[j] * samples[j]
			}
			rms := math.Sqrt(sum / float64(frameSize))
			if rms < 0.01 {
				t.Errorf("Frame %d (t=%.1fs) is too quiet (RMS=%.4f). Sustain failure.", i, float64(i)*0.1, rms)
			}
		}

		// 4. End-to-End Spectral Distance
		t.Log("Verifying spectral distance...")
		target, _, _ := match.LoadWav(targetWav)
		target = match.NormalizeRMS(match.TrimOrPad(target, 44100))
		candidate := match.NormalizeRMS(match.TrimOrPad(samples, 44100))
		dist := match.SpectralDistance(target, candidate)
		t.Logf("Final Functional Spectral Distance: %.6f", dist)
		if dist > 0.1 {
			t.Errorf("Spectral distance too high: %.6f", dist)
		}
	})
}
