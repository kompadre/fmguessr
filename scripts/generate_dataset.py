#!/usr/bin/env python3
"""
Generate probe-parameter training dataset for DX7 NN predictor.

Usage:
    python scripts/generate_dataset.py [--codebook dx7_codebook.bin] [--output dataset.npz] [--variants 50]
    python scripts/generate_dataset.py --codebook dx7_codebook.bin --note-min 36 --note-max 84

This script:
1. Calls the Go-built `generate` binary to render patches at multiple MIDI notes
2. Extracts timbre window (256-bin spectrum) from each rendered WAV
3. Extracts probe features (ADSR, brightness, harmonicity, MIDI note) from each WAV
4. Saves as dataset.npz for PyTorch training
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import struct
import math
import numpy as np
import wave

# Timbre extraction parameters (must match pkg/match/timbre.go)
SAMPLE_RATE = 44100
TIMBRE_SPEC_BINS = 256
TIMBRE_WIN_MS = 300
TIMBRE_SKIP_MS = 80
N_FFT = 1024

# DX7 label dimensions (must match pkg/match/patch.go DX7LabelDims)
DX7_LABEL_DIMS = 53  # Algorithm(1) + 6 ops × 8 params(48) + Feedback(1) + PitchEnv(3)


# --- Probe Feature Extraction (mirrors pkg/match/features.go) ---

def estimate_envelope(samples, sample_rate=SAMPLE_RATE):
    """Extract ADSR envelope matching Go estimateEnvelope."""
    if len(samples) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    frame_size = 256
    n_frames = len(samples) // frame_size
    if n_frames == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute RMS per frame
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = samples[i * frame_size:(i + 1) * frame_size]
        rms[i] = np.sqrt(np.mean(chunk ** 2))

    # Find peak
    peak_idx = np.argmax(rms)
    peak = rms[peak_idx]
    if peak < 1e-10:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Attack: first frame > 10% peak to peak
    attack_threshold = peak * 0.1
    attack_start = 0
    for i in range(n_frames):
        if rms[i] > attack_threshold:
            attack_start = i
            break
    attack_ms = max(0.0, (peak_idx - attack_start) * frame_size / sample_rate * 1000.0)

    # Sound end: last frame > 5% peak
    sound_end_threshold = peak * 0.05
    sound_end_idx = n_frames - 1
    for i in range(n_frames - 1, -1, -1):
        if rms[i] > sound_end_threshold:
            sound_end_idx = i
            break

    # Sustain level: RMS in middle section (20%-70% of active range after peak)
    sustain_start = peak_idx + max(1, int((sound_end_idx - peak_idx) * 0.20))
    sustain_end = peak_idx + max(2, int((sound_end_idx - peak_idx) * 0.70))
    sustain_end = min(sustain_end, n_frames)
    sustain_rms = np.mean(rms[sustain_start:sustain_end]) if sustain_end > sustain_start else 0.0
    sustain_level = sustain_rms / peak

    # Tail level: final 20% of active sound
    tail_start = peak_idx + int((sound_end_idx - peak_idx) * 0.80)
    tail_level = np.mean(rms[tail_start:sound_end_idx + 1]) / peak if sound_end_idx > tail_start else 0.0

    # Key release detection
    release_start_idx = sound_end_idx
    if tail_level >= 0.12 and sustain_level >= 0.20:
        for i in range(sustain_end, sound_end_idx + 1):
            if i < n_frames and rms[i] < sustain_rms * 0.70:
                release_start_idx = i
                break
    else:
        sustain_level = tail_level

    # Decay: peak to sustain threshold
    sustain_threshold = peak * sustain_level * 1.1
    decay_end = peak_idx
    for i in range(peak_idx, n_frames):
        if rms[i] <= sustain_threshold:
            decay_end = i
            break
    decay_ms = max(0.0, (decay_end - peak_idx) * frame_size / sample_rate * 1000.0)

    # Gate: attack start to key release
    gate_frames = release_start_idx - attack_start
    gate_ms = max(1.0, gate_frames * frame_size / sample_rate * 1000.0)

    # Release: key release to sound end
    release_ms = max(0.0, (sound_end_idx - release_start_idx) * frame_size / sample_rate * 1000.0)

    # Percussive sounds: cap gate to 50ms
    if tail_level < 0.12 or sustain_level < 0.15:
        gate_ms = min(gate_ms, 50.0)

    return attack_ms, decay_ms, sustain_level, release_ms, gate_ms


def compute_brightness(samples, sample_rate=SAMPLE_RATE):
    """Compute brightness (high-frequency energy ratio above 2kHz) matching Go computeBrightness."""
    n = len(samples)
    if n < 64:
        return 0.0

    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    windowed = samples[:n] * np.hanning(n)
    spectrum = np.fft.rfft(windowed, fft_size)
    mag = np.abs(spectrum)

    fft_n = len(mag)
    cutoff_bin = max(1, int(2000.0 * fft_size / sample_rate))
    if cutoff_bin >= fft_n:
        cutoff_bin = fft_n - 1

    energy = mag[1:] ** 2
    total_energy = np.sum(energy)
    hf_energy = np.sum(energy[cutoff_bin - 1:])

    if total_energy < 1e-10:
        return 0.0

    ratio = hf_energy / total_energy
    return min(ratio * 5.0, 1.0)


def detect_pitch_frame(frame, sample_rate=SAMPLE_RATE):
    """Detect pitch of a single frame via autocorrelation."""
    n = len(frame)
    if n < 512:
        return 0.0, 0.0

    # Hann window
    windowed = frame * np.hanning(n)

    # Autocorrelation via FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    padded = np.zeros(fft_size)
    padded[:n] = windowed
    spec = np.fft.rfft(padded)
    acorr = np.fft.irfft(spec * np.conj(spec))

    # Only look at lags 20..n/2 (roughly 80Hz to Nyquist)
    min_lag = max(20, int(sample_rate / 1000))  # ~1ms min period
    max_lag = min(n // 2, int(sample_rate / 40))  # ~40Hz max period
    if min_lag >= max_lag or max_lag > len(acorr):
        return 0.0, 0.0

    search = acorr[min_lag:max_lag]
    if len(search) == 0:
        return 0.0, 0.0

    best_lag = min_lag + np.argmax(search)
    if acorr[0] < 1e-10:
        return 0.0, 0.0

    conf = acorr[best_lag] / acorr[0]
    f0 = sample_rate / best_lag
    return f0, min(conf, 1.0)


def detect_pitch(samples, sample_rate=SAMPLE_RATE):
    """Detect pitch via frame-by-frame autocorrelation with energy weighting."""
    if len(samples) < 2048:
        return 0.0, 0.0

    frame_size = 2048
    hop_size = 1024
    detections = []

    for start in range(0, len(samples) - frame_size + 1, hop_size):
        frame = samples[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        f0, conf = detect_pitch_frame(frame, sample_rate)
        if f0 > 20 and conf > 0.2:
            detections.append((f0, conf, rms))

    if not detections:
        return 0.0, 0.0

    max_rms = max(d[2] for d in detections)
    if max_rms < 1e-10:
        return 0.0, 0.0

    # Group by frequency (10% tolerance)
    groups = []
    for f0, conf, rms in detections:
        energy_w = 0.3 + 0.7 * (rms / max_rms)
        score = conf * energy_w
        found = False
        for g in groups:
            if abs(f0 - g['freq']) / max(g['freq'], 1.0) < 0.10:
                g['count'] += 1
                g['total_w'] += score
                g['freq'] = (g['freq'] * (g['count'] - 1) + f0) / g['count']
                if conf > g['best_conf']:
                    g['best_conf'] = conf
                found = True
                break
        if not found:
            groups.append({'freq': f0, 'count': 1, 'best_conf': conf, 'total_w': score})

    best_group = max(groups, key=lambda g: g['total_w'])

    # Sub-harmonic correction
    if best_group['freq'] < 80.0:
        for mult in [2, 3]:
            for g in groups:
                if g is not best_group and abs(g['freq'] - best_group['freq'] * mult) / best_group['freq'] < 0.15:
                    if g['total_w'] > best_group['total_w'] * 0.6:
                        best_group = g

    return best_group['freq'], min(best_group['best_conf'], 1.0)


def estimate_harmonicity(f0, samples, sample_rate=SAMPLE_RATE):
    """Estimate harmonicity (harmonic energy / total energy) matching Go estimateHarmonicity."""
    if f0 < 20 or len(samples) < 256:
        return 0.0

    fft_size = 1
    while fft_size < len(samples):
        fft_size *= 2

    windowed = samples[:len(samples)] * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed, fft_size)
    mag = np.abs(spectrum)

    n = len(mag)
    bin_width = sample_rate / (2.0 * n)

    harmonic_energy = 0.0
    total_energy = 0.0

    for i in range(1, n):
        freq = i * bin_width
        e = mag[i] ** 2
        total_energy += e

        harmonic_num = freq / f0
        nearest_harmonic = max(1, round(harmonic_num))
        harmonic_freq = f0 * nearest_harmonic
        bin_dist = abs(freq - harmonic_freq) / bin_width

        if bin_dist < 2.0:
            harmonic_energy += e

    if total_energy < 1e-10:
        return 0.0

    return harmonic_energy / total_energy


# --- Timbre Spectrum Extraction (unchanged from original) ---

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
    with wave.open(path, 'rb') as w:
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        return samples


def read_labels_bin(path):
    """Read labels.bin written by the Go generate binary.
    
    Format: uint32 count, uint32 dim, then count × dim × float32.
    Returns numpy array of shape (count, dim).
    """
    if not os.path.exists(path):
        return None
    
    with open(path, 'rb') as f:
        count = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.frombuffer(f.read(count * dim * 4), dtype=np.float32).reshape(count, dim)
    
    print(f"  Read labels.bin: {count} samples × {dim} dims")
    return data


def extract_midi_from_filename(filename):
    """Extract MIDI note from WAV filename like 'p0001_v00_n060.wav'."""
    m = re.search(r'_n(\d+)\.wav$', filename)
    if m:
        return int(m.group(1))
    return 60  # fallback to C4


def extract_probe_features(samples, midi_note, sample_rate=SAMPLE_RATE):
    """Extract all9 probe features from rendered audio."""
    attack_ms, decay_ms, sustain_level, release_ms, gate_ms = estimate_envelope(samples, sample_rate)
    brightness = compute_brightness(samples, sample_rate)
    f0, _ = detect_pitch(samples, sample_rate)
    harmonicity = estimate_harmonicity(f0, samples, sample_rate)

    semitone = float(midi_note % 12) / 11.0   # 0-11 → [0,1]
    octave = float(midi_note // 12 - 1) / 8.0  # 0-8 → [0,1]

    return np.array([
        attack_ms / 2000.0,      # normalize to [0,1] roughly (max 2s)
        decay_ms / 5000.0,       # normalize to [0,1] roughly (max 5s)
        sustain_level,           # already [0,1]
        release_ms / 5000.0,     # normalize to [0,1] roughly
        gate_ms / 5000.0,        # normalize to [0,1] roughly
        brightness,              # already [0,1]
        harmonicity,             # already [0,1]
        semitone,                # pitch class 0-11 → [0,1]
        octave,                  # octave 0-8 → [0,1]
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Generate probe-parameter DX7 training dataset')
    parser.add_argument('--codebook', default='dx7_codebook.bin', help='Path to codebook binary')
    parser.add_argument('--output', default='dataset_probe.npz', help='Output dataset file')
    parser.add_argument('--note-min', type=int, default=0, help='Min MIDI note (0=single note mode)')
    parser.add_argument('--note-max', type=int, default=0, help='Max MIDI note (0=single note mode)')
    parser.add_argument('--note', type=int, default=48, help='Single MIDI note (used when note-min/max not set)')
    parser.add_argument('--variants', type=int, default=50, help='Variants per patch')
    parser.add_argument('--tmpdir', default=None, help='Temporary directory for rendered WAVs')
    parser.add_argument('--skip-render', action='store_true', help='Skip Go rendering, process existing WAVs in --tmpdir')
    args = parser.parse_args()

    # Create temp directory for WAV files
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix='fmguessr_probe_')

    if not args.skip_render:
        # Check if generate binary exists
        if not os.path.exists('./generate'):
            print("Building generate binary...")
            subprocess.run(['go', 'build', '-o', 'generate', './cmd/generate/main.go'], check=True)

        # Build render command
        cmd = [
            './generate',
            '--codebook', args.codebook,
            '--outdir', tmpdir,
            '--variants', str(args.variants),
        ]

        if args.note_min > 0 and args.note_max > 0:
            cmd += ['--note-min', str(args.note_min), '--note-max', str(args.note_max)]
            notes = list(range(args.note_min, args.note_max + 1))
            print(f"Rendering {args.variants} variants per patch at MIDI {notes}...")
        else:
            cmd += ['--note', str(args.note)]
            notes = [args.note]
            print(f"Rendering {args.variants} variants per patch at MIDI {args.note}...")

        subprocess.run(cmd, check=True)
    else:
        print(f"Skipping render, processing existing WAVs in {tmpdir}")
        wav_files_all = [f for f in os.listdir(tmpdir) if f.endswith('.wav')]
        note_set = set(extract_midi_from_filename(wf) for wf in wav_files_all)
        notes = sorted(note_set)
        print(f"  Found {len(wav_files_all)} WAVs across {len(notes)} notes: {notes}")

    # Process rendered WAVs
    print("Extracting timbre windows and probe features...")
    timbre_specs = []
    probe_labels = []

    wav_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.wav')])
    print(f"  Processing {len(wav_files)} WAV files...")

    for wav_file in wav_files:
        wav_path = os.path.join(tmpdir, wav_file)
        samples = read_wav_raw(wav_path)
        midi_note = extract_midi_from_filename(wav_file)

        # Extract timbre spectrum (256 bins)
        timbre = extract_timbre_window(samples)
        timbre_specs.append(timbre)

        # Extract probe features (8 values)
        probe = extract_probe_features(samples, midi_note)
        probe_labels.append(probe)

    timbre_specs = np.array(timbre_specs, dtype=np.float32)
    probe_labels = np.array(probe_labels, dtype=np.float32)

    # Read DX7 labels from labels.bin (written by Go generate binary)
    labels_bin_path = os.path.join(tmpdir, 'labels.bin')
    dx7_labels = read_labels_bin(labels_bin_path)

    # Match dx7_labels to WAV files by sorted order
    if dx7_labels is not None and len(dx7_labels) == len(timbre_specs):
        print(f"  DX7 labels shape: {dx7_labels.shape}")
        dx7_names = ['Algorithm', 'Op6.Coarse', 'Op5.Coarse', 'Op4.Coarse', 'Op3.Coarse', 'Op2.Coarse', 'Op1.Coarse',
                      'Op6.Fine', 'Op5.Fine', 'Op4.Fine', 'Op3.Fine', 'Op2.Fine', 'Op1.Fine',
                      'Op6.Level', 'Op5.Level', 'Op4.Level', 'Op3.Level', 'Op2.Level', 'Op1.Level',
                      'Op6.Rate1', 'Op5.Rate1', 'Op4.Rate1', 'Op3.Rate1', 'Op2.Rate1', 'Op1.Rate1',
                      'Op6.Rate2', 'Op5.Rate2', 'Op4.Rate2', 'Op3.Rate2', 'Op2.Rate2', 'Op1.Rate2',
                      'Op6.L3', 'Op5.L3', 'Op4.L3', 'Op3.L3', 'Op2.L3', 'Op1.L3',
                      'Op6.Rate4', 'Op5.Rate4', 'Op4.Rate4', 'Op3.Rate4', 'Op2.Rate4', 'Op1.Rate4',
                      'Feedback', 'Pitch.Rate1', 'Pitch.Lvl1', 'Pitch.Lvl2']
        for i, name in enumerate(dx7_names[:dx7_labels.shape[1]]):
            col = dx7_labels[:, i]
            print(f"    {name:15s}: [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}")
    elif dx7_labels is not None:
        print(f"  WARNING: DX7 labels count ({len(dx7_labels)}) != WAV count ({len(timbre_specs)}), skipping dx7_labels")
        dx7_labels = None

    print(f"  Final dataset: {len(timbre_specs)} samples")
    print(f"  Probe labels shape: {probe_labels.shape}")
    print(f"  Probe label ranges:")
    labels_names = ['AttackMs', 'DecayMs', 'SustainLevel', 'ReleaseMs', 'GateMs', 'Brightness', 'Harmonicity', 'Semitone', 'Octave']
    for i, name in enumerate(labels_names):
        col = probe_labels[:, i]
        print(f"    {name:15s}: [{col.min():.4f}, {col.max():.4f}], mean={col.mean():.4f}")

    # Save as numpy archive
    save_kwargs = dict(
        timbre_specs=timbre_specs,
        probe_labels=probe_labels,
        sample_rate=SAMPLE_RATE,
        timbre_bins=TIMBRE_SPEC_BINS,
        notes=notes,
    )
    if dx7_labels is not None:
        save_kwargs['dx7_labels'] = dx7_labels

    np.savez_compressed(args.output, **save_kwargs)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Dataset saved to {args.output} ({file_size:.1f} MB)")

    # Clean up temp directory
    if args.tmpdir is None:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
