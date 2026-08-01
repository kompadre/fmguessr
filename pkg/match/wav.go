package match

import (
	"bytes"
	"fmt"
	"math"
	"os"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

// DecodeWavBytes decodes WAV file bytes and returns mono float64 samples [-1, 1].
func DecodeWavBytes(data []byte) ([]float64, int, error) {
	r := bytes.NewReader(data)
	dec := wav.NewDecoder(r)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decode wav bytes: %w", err)
	}

	sampleRate := buf.Format.SampleRate
	numChans := buf.Format.NumChannels

	mono := make([]float64, len(buf.Data)/numChans)
	bitDepth := dec.BitDepth
	maxVal := float64(int(1)<<(bitDepth-1) - 1)
	if bitDepth == 8 {
		maxVal = 127.0
	}

	for i := 0; i < len(mono); i++ {
		sum := 0.0
		for ch := 0; ch < numChans; ch++ {
			sample := buf.Data[i*numChans+ch]
			if bitDepth == 8 {
				sum += float64(sample-128) / 128.0
			} else {
				sum += float64(sample) / maxVal
			}
		}
		mono[i] = sum / float64(numChans)
	}

	return mono, sampleRate, nil
}

// LoadWav reads a WAV file and returns mono float64 samples normalized to [-1, 1].
func LoadWav(path string) ([]float64, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open wav: %w", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decode wav: %w", err)
	}

	sampleRate := buf.Format.SampleRate
	numChans := buf.Format.NumChannels

	mono := make([]float64, len(buf.Data)/numChans)
	bitDepth := dec.BitDepth
	maxVal := float64(int(1)<<(bitDepth-1) - 1)
	if bitDepth == 8 {
		maxVal = 127.0
	}

	for i := 0; i < len(mono); i++ {
		sum := 0.0
		for ch := 0; ch < numChans; ch++ {
			sample := buf.Data[i*numChans+ch]
			if bitDepth == 8 {
				sum += float64(sample-128) / 128.0
			} else {
				sum += float64(sample) / maxVal
			}
		}
		mono[i] = sum / float64(numChans)
	}

	return mono, sampleRate, nil
}

// SaveWav encodes float64 PCM samples [-1, 1] to a 16-bit mono WAV file.
func SaveWav(path string, samples []float64, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create wav file: %w", err)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, 1, 1)

	intBuf := make([]int, len(samples))
	for i, s := range samples {
		v := s
		if v > 1.0 {
			v = 1.0
		}
		if v < -1.0 {
			v = -1.0
		}
		intBuf[i] = int(v * 32767.0)
	}

	buf := &audio.IntBuffer{
		Format: &audio.Format{
			NumChannels: 1,
			SampleRate:  sampleRate,
		},
		Data: intBuf,
	}

	if err := enc.Write(buf); err != nil {
		return fmt.Errorf("write wav: %w", err)
	}
	if err := enc.Close(); err != nil {
		return fmt.Errorf("close wav encoder: %w", err)
	}
	return nil
}

// TrimOrPad trims or zero-pads samples to exactly targetLen.
func TrimOrPad(samples []float64, targetLen int) []float64 {
	out := make([]float64, targetLen)
	n := len(samples)
	if n > targetLen {
		n = targetLen
	}
	copy(out, samples[:n])
	return out
}

// NormalizePeak scales samples so the peak absolute value is 1.0.
func NormalizePeak(samples []float64) []float64 {
	peak := 0.0
	for _, s := range samples {
		if v := math.Abs(s); v > peak {
			peak = v
		}
	}
	if peak < 1e-10 {
		return samples
	}
	out := make([]float64, len(samples))
	for i, s := range samples {
		out[i] = s / peak
	}
	return out
}

// CalcRMS computes the un-normalized root mean square energy of a signal.
func CalcRMS(samples []float64) float64 {
	if len(samples) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range samples {
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(samples)))
}

// NormalizeRMS scales samples so the RMS level is 0.1.
// This preserves relative dynamic range better than peak normalization
// for loud samples with sharp transients.
func NormalizeRMS(samples []float64) []float64 {
	sum := 0.0
	for _, v := range samples {
		sum += v * v
	}
	rms := math.Sqrt(sum / float64(len(samples)))
	if rms < 1e-10 {
		return samples
	}
	out := make([]float64, len(samples))
	scale := 0.1 / rms
	for i, v := range samples {
		out[i] = v * scale
	}
	return out
}