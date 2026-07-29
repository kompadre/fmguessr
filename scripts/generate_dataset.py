#!/usr/bin/env python3
"""
Generate training dataset for DX7 parameter estimation CNN.

Usage:
    python scripts/generate_dataset.py [--codebook dx7_codebook.bin] [--output dataset.npz] [--note 48] [--variants 50]

This script:
1. Loads the Go-built `generate` binary to render codebook patches
2. Processes rendered WAV files into mel spectrograms
3. Pairs spectrograms with normalized DX7 parameters
4. Saves as dataset.npz for PyTorch training
"""

import argparse
import os
import subprocess
import sys
import tempfile
import struct
import numpy as np

# Mel spectrogram parameters (must match pkg/match/spectrogram.go)
SAMPLE_RATE = 44100
N_MELS = 128
N_FRAMES = 128
HOP_LENGTH = 512
WIN_LENGTH = 2048
F_MIN = 20.0
F_MAX = 8000.0


def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sample_rate):
    """Compute mel filterbank matrix."""
    f_min = F_MIN
    f_max = F_MAX

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        start = bin_points[m]
        center = bin_points[m + 1]
        end = bin_points[m + 2]

        for k in range(start, center):
            if center > start:
                filterbank[m, k] = (k - start) / (center - start)
        for k in range(center, end):
            if end > center:
                filterbank[m, k] = (end - k) / (end - center)

    return filterbank


def compute_mel_spectrogram(samples, sample_rate=SAMPLE_RATE):
    """Compute 128x128 mel spectrogram from audio samples."""
    import numpy as np

    # Ensure we have enough samples
    if len(samples) < WIN_LENGTH:
        samples = np.pad(samples, (0, WIN_LENGTH - len(samples)))
    if len(samples) > N_FRAMES * HOP_LENGTH + WIN_LENGTH:
        samples = samples[:N_FRAMES * HOP_LENGTH + WIN_LENGTH]

    # STFT
    n_fft = WIN_LENGTH
    hop = HOP_LENGTH
    window = np.hanning(WIN_LENGTH)

    n_frames = (len(samples) - WIN_LENGTH) // hop + 1
    spectrogram = np.zeros((N_MELS, n_frames))

    filterbank = mel_filterbank(N_MELS, n_fft, SAMPLE_RATE)

    for i in range(n_frames):
        start = i * hop
        frame = samples[start:start + WIN_LENGTH] * window
        spectrum = np.abs(np.fft.rfft(frame))
        mel_spec = filterbank @ spectrum
        mel_spec = np.log(mel_spec + 1e-5)
        spectrogram[:, i] = mel_spec

    # Pad or truncate to N_FRAMES
    if spectrogram.shape[1] < N_FRAMES:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, N_FRAMES - spectrogram.shape[1])))
    else:
        spectrogram = spectrogram[:, :N_FRAMES]

    # Normalize to [0, 1]
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if spec_max - spec_min > 1e-8:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        spectrogram = np.zeros_like(spectrogram)

    return spectrogram


def read_wav_raw(path):
    """Read a mono 16-bit WAV file and return float64 samples in [-1, 1]."""
    import wave
    with wave.open(path, 'rb') as w:
        n_frames = w.getnframes()
        sample_width = w.getsampwidth()
        raw = w.readframes(n_frames)

        if sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        elif sample_width == 3:
            # 24-bit audio
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
            # Reconstruct 24-bit signed
            samples = (samples[:, 0] + samples[:, 1] * 256 + samples[:, 2] * 65536)
            samples = np.where(samples >= 8388608, samples - 16777216, samples)
            samples = samples / 8388608.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        return samples


def normalize_params(algo, ops, feedback):
    """Normalize DX7 parameters to [0, 1] range for training."""
    normalized = np.zeros(50, dtype=np.float32)

    # Algorithm (0-31) -> [0, 1]
    normalized[0] = algo / 31.0

    # Per-operator params (6 ops × 8 params = 48 values)
    # Layout: [Mode, FreqCoarse, FreqFine, OutputLevel, Rate1, Rate2, Level3, Rate4] × 6
    for op_idx in range(6):
        base = 1 + op_idx * 8
        normalized[base + 0] = ops[op_idx][0]  # Mode (0 or 1, already [0,1])
        normalized[base + 1] = ops[op_idx][1] / 31.0  # FreqCoarse
        normalized[base + 2] = ops[op_idx][2] / 99.0  # FreqFine
        normalized[base + 3] = ops[op_idx][3] / 99.0  # OutputLevel
        normalized[base + 4] = ops[op_idx][4] / 99.0  # Rate1
        normalized[base + 5] = ops[op_idx][5] / 99.0  # Rate2
        normalized[base + 6] = ops[op_idx][6] / 99.0  # Level3
        normalized[base + 7] = ops[op_idx][7] / 99.0  # Rate4

    # Feedback (0-7) -> [0, 1]
    normalized[49] = feedback / 7.0

    return normalized


def main():
    parser = argparse.ArgumentParser(description='Generate DX7 training dataset')
    parser.add_argument('--codebook', default='dx7_codebook.bin', help='Path to codebook binary')
    parser.add_argument('--output', default='dataset.npz', help='Output dataset file')
    parser.add_argument('--note', type=int, default=48, help='MIDI note to render (default: 48 = C3)')
    parser.add_argument('--variants', type=int, default=50, help='Variants per patch (default: 50)')
    parser.add_argument('--tmpdir', default=None, help='Temporary directory for rendered WAVs')
    args = parser.parse_args()

    # Check if generate binary exists
    if not os.path.exists('./generate'):
        print("Building generate binary...")
        subprocess.run(['go', 'build', '-o', 'generate', './cmd/generate/main.go'], check=True)

    # Create temp directory for WAV files
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix='fmguessr_')
    os.makedirs(tmpdir, exist_ok=True)

    print(f"Rendering {args.variants} variants per patch...")

    # Run Go binary to render all patches
    cmd = [
        './generate',
        '--codebook', args.codebook,
        '--outdir', tmpdir,
        '--note', str(args.note),
        '--variants', str(args.variants),
    ]
    subprocess.run(cmd, check=True)

    # Process rendered WAVs and labels
    print("Processing rendered audio into spectrograms...")
    spectrograms = []
    labels = []

    # Read labels file
    labels_path = os.path.join(tmpdir, 'labels.bin')
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found")
        sys.exit(1)

    with open(labels_path, 'rb') as f:
        # Read header: n_samples, n_params
        n_samples = struct.unpack('I', f.read(4))[0]
        n_params = struct.unpack('I', f.read(4))[0]
        print(f"  {n_samples} samples, {n_params} parameters each")

        # Read all labels
        all_labels = np.frombuffer(f.read(n_samples * n_params * 4), dtype=np.float32)
        all_labels = all_labels.reshape(n_samples, n_params)

    # Process each WAV file
    wav_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.wav')])
    print(f"  Processing {len(wav_files)} WAV files...")

    for wav_file in wav_files:
        wav_path = os.path.join(tmpdir, wav_file)
        samples = read_wav_raw(wav_path)

        # Compute mel spectrogram
        spec = compute_mel_spectrogram(samples)
        spectrograms.append(spec)

    spectrograms = np.array(spectrograms, dtype=np.float32)

    # Ensure we have matching counts
    min_count = min(len(spectrograms), len(all_labels))
    spectrograms = spectrograms[:min_count]
    all_labels = all_labels[:min_count]

    print(f"  Final dataset: {min_count} samples")

    # Save as numpy archive
    np.savez_compressed(args.output,
        spectrograms=spectrograms,
        labels=all_labels,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_frames=N_FRAMES,
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Dataset saved to {args.output} ({file_size:.1f} MB)")

    # Clean up temp directory
    if args.tmpdir is None:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
