package match

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

var (
	sessionMagic   = [4]byte{'T', 'R', 'A', 'J'}
	sessionVersion = uint8(1)
)

// SessionData holds pre-computed trajectory analysis for a target audio clip.
type SessionData struct {
	AudioHash   [32]byte
	Trajectory  *TrajectorySpec
	PhaseBounds PhaseBounds
	LFOSpectrum []float64
}

// AudioHash computes SHA256 of normalized float64 samples for deduplication.
func AudioHash(samples []float64) [32]byte {
	h := sha256.New()
	buf := make([]byte, 8)
	for _, s := range samples {
		binary.LittleEndian.PutUint64(buf, math.Float64bits(s))
		h.Write(buf)
	}
	var hash [32]byte
	copy(hash[:], h.Sum(nil))
	return hash
}

// SaveSession writes trajectory analysis to a binary session file atomically.
// Uses temp file + os.Rename to prevent partial reads by concurrent jobs.
func SaveSession(path string, data *SessionData) error {
	if data.Trajectory == nil {
		return fmt.Errorf("save session: trajectory is nil")
	}

	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("create session file: %w", err)
	}
	defer os.Remove(tmpPath) // cleanup on early return

	// Header: 4 magic + 1 version + 4×uint32 (embDim, timeSteps, attackEnd, sustainEnd) + 4 lfoBins + 32 hash
	if _, err := f.Write(sessionMagic[:]); err != nil {
		return err
	}
	if _, err := f.Write([]byte{sessionVersion}); err != nil {
		return err
	}

	var hdr [24]byte
	binary.LittleEndian.PutUint32(hdr[0:4], uint32(data.Trajectory.EmbDim))
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(data.Trajectory.TimeSteps))
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(data.PhaseBounds.AttackEnd))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(data.PhaseBounds.SustainEnd))
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(len(data.LFOSpectrum)))
	binary.LittleEndian.PutUint32(hdr[20:24], uint32(0)) // reserved
	if _, err := f.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := f.Write(data.AudioHash[:]); err != nil {
		return err
	}

	// Embeddings: N × EmbDim × float32 (row-major)
	floatBuf := make([]byte, 4)
	for _, vec := range data.Trajectory.Embeddings {
		for _, val := range vec {
			binary.LittleEndian.PutUint32(floatBuf, math.Float32bits(val))
			if _, err := f.Write(floatBuf); err != nil {
				return err
			}
		}
	}

	// LFO Spectrum: bins × float32
	for _, val := range data.LFOSpectrum {
		binary.LittleEndian.PutUint32(floatBuf, math.Float32bits(float32(val)))
		if _, err := f.Write(floatBuf); err != nil {
			return err
		}
	}

	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

// LoadSession reads trajectory analysis from a binary session file.
func LoadSession(path string) (*SessionData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open session file: %w", err)
	}
	defer f.Close()

	// Read magic
	var magic [4]byte
	if _, err := f.Read(magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != sessionMagic {
		return nil, fmt.Errorf("invalid session magic: %x", magic)
	}

	// Read version
	ver := make([]byte, 1)
	if _, err := f.Read(ver); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if ver[0] != sessionVersion {
		return nil, fmt.Errorf("unsupported session version: %d", ver[0])
	}

	// Read header
	var hdr [24]byte
	if _, err := f.Read(hdr[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	embDim := int(binary.LittleEndian.Uint32(hdr[0:4]))
	timeSteps := int(binary.LittleEndian.Uint32(hdr[4:8]))
	attackEnd := int(binary.LittleEndian.Uint32(hdr[8:12]))
	sustainEnd := int(binary.LittleEndian.Uint32(hdr[12:16]))
	lfoBins := int(binary.LittleEndian.Uint32(hdr[16:20]))

	// Read audio hash
	var hash [32]byte
	if _, err := f.Read(hash[:]); err != nil {
		return nil, fmt.Errorf("read hash: %w", err)
	}

	// Read embeddings
	embeddings := make([][]float32, timeSteps)
	floatBuf := make([]byte, 4)
	for i := 0; i < timeSteps; i++ {
		embeddings[i] = make([]float32, embDim)
		for j := 0; j < embDim; j++ {
			if _, err := f.Read(floatBuf); err != nil {
				return nil, fmt.Errorf("read embedding[%d][%d]: %w", i, j, err)
			}
			embeddings[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(floatBuf))
		}
	}

	// Read LFO spectrum
	lfoSpectrum := make([]float64, lfoBins)
	for i := 0; i < lfoBins; i++ {
		if _, err := f.Read(floatBuf); err != nil {
			return nil, fmt.Errorf("read lfo[%d]: %w", i, err)
		}
		lfoSpectrum[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(floatBuf)))
	}

	return &SessionData{
		AudioHash: hash,
		Trajectory: &TrajectorySpec{
			Embeddings: embeddings,
			TimeSteps:  timeSteps,
			EmbDim:     embDim,
		},
		PhaseBounds: PhaseBounds{
			AttackEnd:  attackEnd,
			SustainEnd: sustainEnd,
		},
		LFOSpectrum: lfoSpectrum,
	}, nil
}
