#!/usr/bin/env python3
"""
Classify DX7 patches from a codebook using PANNs CNN14 ONNX model.

Renders each patch to audio via ./play, classifies with the ONNX model,
and outputs patch name + curated tag probabilities.

Usage:
    python scripts/patch_classifier/classify.py \
        --codebook data/dx7_codebook.bin \
        --model panns_inference.onnx \
        --play-bin ./play

Output format:
    Patch 0000: E.PIANO 1    → piano=0.82, synth=0.45, bell=0.31
    Patch 0001: BASS 1       → bass=0.91, synth=0.22
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# ──────────────────────────────────────────────────────────────────
# AudioSet class indices → curated tag mapping
# ──────────────────────────────────────────────────────────────────

# Curated tag dictionary: tag name → list of AudioSet class indices
TAG_MAP = {
    "piano":  [0, 1],            # Piano, Musical keyboard
    "strings": [35, 36, 37, 38], # Violin/fiddle, Cello, Double bass, Harp
    "wind":   [42, 43, 44, 45],  # Flute, Clarinet, Oboe, Saxophone
    "brass":  [46, 47, 48],      # Trumpet, Trombone, French horn
    "guitar": [25, 26],          # Acoustic guitar, Electric guitar
    "bass":   [27],              # Bass guitar
    "synth":  [137],             # Synthesizer
    "organ":  [13],              # Organ
    "bell":   [39, 40],          # Bell, Chime
    "drum":   [128, 129, 130],   # Drum kit, Snare drum, Bass drum
    "choir":  [6, 7],            # Choir, Singing
}

# Build reverse mapping: AudioSet index → tag name
ASET_TO_TAG = {}
for tag, indices in TAG_MAP.items():
    for idx in indices:
        ASET_TO_TAG[idx] = tag


# ──────────────────────────────────────────────────────────────────
# Codebook parsing (DXCB v1 format)
# ──────────────────────────────────────────────────────────────────

def read_codebook(path):
    """Read DXCB codebook. Returns (entries, version) where each entry is a dict."""
    with open(path, 'rb') as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"Codebook too small: {len(header)} bytes")
        magic = header[:4]
        if magic != b'DXCB':
            raise ValueError(f"Not a DXCB codebook: magic={magic}")
        version = struct.unpack('<I', header[4:8])[0]
        count = struct.unpack('<I', header[8:12])[0]
        print(f"Codebook: {count} entries, version {version}")

        entries = []
        for i in range(count):
            name_raw = f.read(16)
            patch = f.read(128)
            fingerprint = f.read(256)
            if len(patch) < 128 or len(fingerprint) < 256:
                raise ValueError(f"Entry {i}: truncated (name={len(name_raw)}, patch={len(patch)}, fp={len(fingerprint)})")
            name = name_raw.rstrip(b'\x00 ').decode('ascii', errors='replace').strip()
            entries.append({
                'index': i,
                'name': name,
                'patch': patch,
                'fingerprint': fingerprint,
            })

    print(f"Loaded {len(entries)} patches from {path}")
    return entries


# ──────────────────────────────────────────────────────────────────
# WAV loading (PCM int16, mono)
# ──────────────────────────────────────────────────────────────────

def load_wav(path):
    """Load a WAV file and return (samples_float32, sample_rate)."""
    with open(path, 'rb') as f:
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a WAV file: {path}")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a WAV file: {path}")
        fmt = f.read(4)
        chunk_size = struct.unpack('<I', f.read(4))[0]
        audio_format = struct.unpack('<H', f.read(2))[0]
        num_channels = struct.unpack('<H', f.read(2))[0]
        sample_rate = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # byte rate
        f.read(2)  # block align
        bits_per_sample = struct.unpack('<H', f.read(2))[0]
        f.read(chunk_size - 16)  # skip remaining fmt chunk

        # Find data chunk
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                raise ValueError("Data chunk not found")
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'data':
                break
            f.read(chunk_size)

        raw_data = f.read(chunk_size)

    # Convert to float32
    if bits_per_sample == 16:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        if audio_format == 3:  # IEEE float
            samples = np.frombuffer(raw_data, dtype=np.float32)
        else:
            samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

    # Convert stereo to mono
    if num_channels == 2:
        samples = (samples[0::2] + samples[1::2]) / 2.0

    return samples, sample_rate


def resample(audio, src_rate, dst_rate):
    """Simple linear interpolation resampling."""
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    n_out = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# PANNs CNN14 inference
# ──────────────────────────────────────────────────────────────────

def load_model(model_path):
    """Load PANNs CNN14 ONNX model."""
    import onnxruntime as ort
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"  Input: {input_name} shape={input_shape}")
    print(f"  Output: {output_name}")
    return session, input_name, output_name


def classify_audio(session, input_name, output_name, audio_32k):
    """Classify audio using PANNs CNN14. Returns 527 class probabilities."""
    # PANNs expects (1, samples) float32, mono, 32kHz
    # Use 1-second window (32000 samples) — take center of the audio
    n_needed = 32000
    if len(audio_32k) < n_needed:
        # Pad with zeros if too short
        audio_32k = np.pad(audio_32k, (0, n_needed - len(audio_32k)))
    elif len(audio_32k) > n_needed:
        # Take center
        start = (len(audio_32k) - n_needed) // 2
        audio_32k = audio_32k[start:start + n_needed]

    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio_32k))
    if peak > 0:
        audio_32k = audio_32k / peak

    # Reshape for batch dimension
    audio_32k = audio_32k.reshape(1, -1).astype(np.float32)

    # Run inference
    output = session.run([output_name], {input_name: audio_32k})[0]

    # Apply sigmoid (PANNs output is raw logits for multi-label)
    probs = 1.0 / (1.0 + np.exp(-output[0]))

    return probs


def map_to_tags(probs, threshold=0.3, top_k=5):
    """Map AudioSet probabilities to curated tags."""
    tag_scores = {}
    for aset_idx, prob in enumerate(probs):
        if prob >= threshold and aset_idx in ASET_TO_TAG:
            tag = ASET_TO_TAG[aset_idx]
            if tag not in tag_scores or prob > tag_scores[tag]:
                tag_scores[tag] = float(prob)

    # Sort by probability descending, take top k
    sorted_tags = sorted(tag_scores.items(), key=lambda x: -x[1])[:top_k]
    return sorted_tags


# ──────────────────────────────────────────────────────────────────
# Render via ./play
# ──────────────────────────────────────────────────────────────────

def render_patch(play_bin, codebook_path, slot, note="C3"):
    """Render a single patch to a temp WAV file using ./play.
    Returns path to WAV file.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [play_bin, '-o', tmp_path, codebook_path, str(slot), note]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  Warning: ./play failed for slot {slot}: {result.stderr.strip()}", file=sys.stderr)
            return None
        return tmp_path
    except subprocess.TimeoutExpired:
        print(f"  Warning: ./play timed out for slot {slot}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: ./play error for slot {slot}: {e}", file=sys.stderr)
        return None


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Classify DX7 patches from a codebook using PANNs CNN14 ONNX model'
    )
    parser.add_argument('--codebook', required=True, help='Path to DXCB codebook .bin file')
    parser.add_argument('--model', required=True, help='Path to PANNs CNN14 ONNX model')
    parser.add_argument('--play-bin', default='./play', help='Path to ./play binary (default: ./play)')
    parser.add_argument('--note', default='C3', help='MIDI note for rendering (default: C3)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Tag probability threshold (default: 0.3)')
    parser.add_argument('--top-k', type=int, default=5, help='Max tags per patch (default: 5)')
    parser.add_argument('--start', type=int, default=0, help='Start at patch index (default: 0)')
    parser.add_argument('--end', type=int, default=None, help='End at patch index (default: all)')
    args = parser.parse_args()

    # Load codebook
    entries = read_codebook(args.codebook)

    # Load ONNX model
    if not os.path.isfile(args.model):
        print(f"Error: ONNX model not found: {args.model}", file=sys.stderr)
        print(f"Download panns_inference.onnx to the project root and try again.", file=sys.stderr)
        sys.exit(1)
    session, input_name, output_name = load_model(args.model)

    # Verify play binary exists
    if not os.path.isfile(args.play_bin):
        print(f"Error: play binary not found: {args.play_bin}", file=sys.stderr)
        sys.exit(1)

    # Classify each patch
    end = args.end if args.end is not None else len(entries)
    total = end - args.start
    print(f"\nClassifying patches {args.start}..{end-1} ({total} patches, note={args.note})")
    print("-" * 70)

    start_time = time.time()
    for i in range(args.start, min(end, len(entries))):
        entry = entries[i]
        name = entry['name'] or f'PATCH_{i:04d}'

        # Render patch to WAV
        wav_path = render_patch(args.play_bin, args.codebook, i, args.note)
        if wav_path is None:
            print(f"Patch {i:04d}: {name:16s} → (render failed)")
            continue

        try:
            # Load WAV
            audio, src_rate = load_wav(wav_path)

            # Resample to 32kHz for PANNs
            audio_32k = resample(audio, src_rate, 32000)

            # Classify
            probs = classify_audio(session, input_name, output_name, audio_32k)

            # Map to tags
            tags = map_to_tags(probs, threshold=args.threshold, top_k=args.top_k)

            # Format output
            tag_str = ', '.join(f'{tag}={prob:.2f}' for tag, prob in tags)
            print(f"Patch {i:04d}: {name:16s} → {tag_str or '(no tags)'}")

        except Exception as e:
            print(f"Patch {i:04d}: {name:16s} → (error: {e})", file=sys.stderr)
        finally:
            # Clean up temp WAV
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    elapsed = time.time() - start_time
    print(f"\nDone: {total} patches classified in {elapsed:.1f}s "
          f"({elapsed/max(total,1):.1f}s per patch)")


if __name__ == '__main__':
    main()
