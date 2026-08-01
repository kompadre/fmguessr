#!/usr/bin/env python3
"""
Generate timbre-focused training dataset for DX7 parameter estimation CNN.

Usage:
    python scripts/generate_dataset.py [--codebook dx7_codebook.bin] [--output dataset.npz] [--note 48] [--variants 50]

This script:
1. Loads the Go-built `generate` binary to render pure sustain samples
2. Extracts timbre window from each sample (skip attack, select stable segment)
3. Computes timbre spectrum (256-bin compressed median STFT)
4. Pairs timbre spectra with timbre-only labels (envelope params neutral)
5. Saves as dataset.npz for PyTorch training
"""

import argparse
import os
import subprocess
import sys
import tempfile
import struct
import math
import numpy as np

# Timbre extraction parameters (must match pkg/match/timbre.go)
SAMPLE_RATE = 44100
TIMBRE_SPEC_BINS = 256
TIMBRE_WIN_MS = 300
TIMBRE_SKIP_MS = 80
N_FFT = 1024


def extract_timbre_window(samples, sample_rate=SAMPLE_RATE):
    """Extract timbre window matching Go ExtractTimbreWindow exactly."""
    if len(samples) < 1024:
        return np.zeros(TIMBRE_SPEC_BINS)

    skip_samples = int(sample_rate * TIMBRE_SKIP_MS / 1000)
    win_samples = int(sample_rate * TIMBRE_WIN_MS / 1000)

    if skip_samples >= len(samples):
        skip_samples = 0
    if skip_samples + win_samples > len(samples):
        win_samples = len(samples) - skip_samples
    if win_samples < 1024:
        return np.zeros(TIMBRE_SPEC_BINS)

    sustain = samples[skip_samples:skip_samples + win_samples]

    # Compute spectral flux per frame
    flux_win = 1024
    flux_hop = 512
    n_flux_frames = max(1, (len(sustain) - flux_win) // flux_hop)

    fluxes = np.zeros(n_flux_frames)
    prev_mag = None
    for i in range(n_flux_frames):
        start = i * flux_hop
        frame = sustain[start:start + flux_win]
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        if prev_mag is not None:
            diff = mag - prev_mag
            fluxes[i] = np.sqrt(np.sum(diff[diff > 0] ** 2))
        prev_mag = mag

    # Find window with minimum flux
    best_start = 0
    best_flux = float('inf')
    for i in range(0, len(sustain) - flux_win, flux_hop):
        total_flux = 0.0
        count = 0
        for j in range(i, min(i + win_samples, len(sustain) - flux_win), flux_hop):
            idx = j // flux_hop
            if idx < len(fluxes):
                total_flux += fluxes[idx]
                count += 1
        if count > 0:
            total_flux /= count
        if total_flux < best_flux:
            best_flux = total_flux
            best_start = i

    stable = sustain[best_start:best_start + win_samples]

    # Normalize frame energies
    frame_len = 512
    n_frames = max(1, len(stable) // frame_len)
    norm_frames = []

    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(stable))
        frame = stable[start:end]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1e-10:
            rms = 1e-10
        norm_frames.append(frame / rms)

    # Compute temporal median STFT
    half_win = N_FFT // 2 + 1
    window = np.hanning(N_FFT)
    all_mags = []

    for frame in norm_frames:
        if len(frame) < N_FFT:
            padded = np.zeros(N_FFT)
            padded[:len(frame)] = frame
            frame = padded
        fft_in = frame * window
        spectrum = np.fft.rfft(fft_in)
        mag = np.abs(spectrum)
        all_mags.append(mag)

    # Median across frames
    all_mags_arr = np.array(all_mags)
    median_spec = np.median(all_mags_arr, axis=0)

    # Compress to TIMBRE_SPEC_BINS (256)
    bins_per_output = max(1, half_win // TIMBRE_SPEC_BINS)
    result = np.zeros(TIMBRE_SPEC_BINS)
    for i in range(TIMBRE_SPEC_BINS):
        start = i * bins_per_output
        end = min((i + 1) * bins_per_output, half_win)
        if end > start:
            result[i] = np.mean(median_spec[start:end])

    # Normalize to [0, 1]
    spec_min = result.min()
    spec_max = result.max()
    if spec_max - spec_min > 1e-8:
        result = (result - spec_min) / (spec_max - spec_min)

    return result


def read_wav_raw(path):
    """Read a mono 16-bit WAV file and return float64 samples in [-1, 1]."""
    import wave
    with wave.open(path, 'rb') as w:
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        return samples


def main():
    parser = argparse.ArgumentParser(description='Generate timbre-focused DX7 training dataset')
    parser.add_argument('--codebook', default='dx7_codebook.bin', help='Path to codebook binary')
    parser.add_argument('--output', default='dataset_timbre.npz', help='Output dataset file')
    parser.add_argument('--note', type=int, default=48, help='MIDI note to render (default: 48 = C3)')
    parser.add_argument('--variants', type=int, default=50, help='Variants per patch')
    parser.add_argument('--tmpdir', default=None, help='Temporary directory for rendered WAVs')
    args = parser.parse_args()

    # Check if generate binary exists
    if not os.path.exists('./generate'):
        print("Building generate binary...")
        subprocess.run(['go', 'build', '-o', 'generate', './cmd/generate/main.go'], check=True)

    # Create temp directory for WAV files
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix='fmguessr_timbre_')

    print(f"Rendering {args.variants} variants per patch at MIDI {args.note}...")

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
    print("Extracting timbre windows and processing labels...")
    timbre_specs = []
    labels = []

    # Read labels file
    labels_path = os.path.join(tmpdir, 'labels.bin')
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found")
        sys.exit(1)

    with open(labels_path, 'rb') as f:
        n_samples = struct.unpack('I', f.read(4))[0]
        n_params = struct.unpack('I', f.read(4))[0]
        print(f"  {n_samples} samples, {n_params} parameters each")

        all_labels = np.frombuffer(f.read(n_samples * n_params * 4), dtype=np.float32)
        all_labels = all_labels.reshape(n_samples, n_params)

    # Process each WAV file
    wav_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.wav')])
    print(f"  Processing {len(wav_files)} WAV files...")

    for wav_file in wav_files:
        wav_path = os.path.join(tmpdir, wav_file)
        samples = read_wav_raw(wav_path)

        # Extract timbre window
        timbre = extract_timbre_window(samples)
        timbre_specs.append(timbre)

    timbre_specs = np.array(timbre_specs, dtype=np.float32)

    # Ensure matching counts
    min_count = min(len(timbre_specs), len(all_labels))
    timbre_specs = timbre_specs[:min_count]
    all_labels = all_labels[:min_count]

    print(f"  Final dataset: {min_count} samples")

    # Save as numpy archive
    np.savez_compressed(args.output,
        timbre_specs=timbre_specs,
        labels=all_labels,
        sample_rate=SAMPLE_RATE,
        timbre_bins=TIMBRE_SPEC_BINS,
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Dataset saved to {args.output} ({file_size:.1f} MB)")

    # Clean up temp directory
    if args.tmpdir is None:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
