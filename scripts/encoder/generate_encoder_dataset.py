#!/usr/bin/env python3
"""
Generate training dataset for the 1D ResNet audio encoder.

Renders DX7 patches at multiple MIDI notes into raw WAV files.
Window extraction and augmentation happen on-the-fly during training.

Usage:
    python scripts/encoder/generate_encoder_dataset.py --codebook dx7_codebook.bin --outdir encoder_wavs
    python scripts/encoder/generate_encoder_dataset.py --codebook dx7_codebook.bin --outdir encoder_wavs --note-min 36 --note-max 84

Output: directory of WAV files + manifest.csv
    encoder_wavs/
        manifest.csv        — patch_id, midi_note, wav_path
        p000_n048.wav
        p000_n060.wav
        ...
"""

import argparse
import csv
import os
import subprocess
import sys
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description='Generate 1D ResNet encoder training WAVs')
    parser.add_argument('--codebook', default='dx7_codebook.bin', help='Path to codebook binary')
    parser.add_argument('--outdir', default='encoder_wavs', help='Output directory for WAV files')
    parser.add_argument('--note-min', type=int, default=48, help='Min MIDI note (default: 48 = C3)')
    parser.add_argument('--note-max', type=int, default=72, help='Max MIDI note (default: 72 = C5)')
    parser.add_argument('--variants', type=int, default=1, help='Variants per patch per note')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    renders_dir = os.path.join(script_dir, 'renders')

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)

    # Build generate binary if needed (in repo root)
    generate_bin = os.path.join(repo_root, 'generate')
    if not os.path.exists(generate_bin):
        print("Building generate binary...")
        subprocess.run(['go', 'build', '-o', 'generate', './cmd/generate/main.go'],
                       check=True, cwd=repo_root)

    # Render using Go binary into renders staging dir
    cmd = [
        generate_bin,
        '--codebook', args.codebook,
        '--outdir', renders_dir,
        '--note-min', str(args.note_min),
        '--note-max', str(args.note_max),
        '--variants', str(args.variants),
    ]

    notes = list(range(args.note_min, args.note_max + 1))
    print(f"Rendering patches at MIDI {args.note_min}-{args.note_max} ({len(notes)} notes)...")
    subprocess.run(cmd, check=True)

    # Move WAVs from staging to outdir and build manifest
    manifest = []
    wav_files = sorted([f for f in os.listdir(renders_dir) if f.endswith('.wav')])

    print(f"Moving {len(wav_files)} WAVs to {args.outdir}...")
    for wav_file in wav_files:
        src = os.path.join(renders_dir, wav_file)
        dst = os.path.join(args.outdir, wav_file)
        shutil.move(src, dst)

        pid_m = re.search(r'p(\d+)_', wav_file)
        note_m = re.search(r'_n(\d+)\.wav$', wav_file)
        patch_id = int(pid_m.group(1)) if pid_m else 0
        midi_note = int(note_m.group(1)) if note_m else 60

        manifest.append((patch_id, midi_note, wav_file))

    # Write manifest
    manifest_path = os.path.join(args.outdir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patch_id', 'midi_note', 'wav_file'])
        for row in manifest:
            writer.writerow(row)

    n_patches = len(set(r[0] for r in manifest))
    print(f"\nDone! {manifest_path}")
    print(f"  WAVs: {len(manifest)}")
    print(f"  Patches: {n_patches}")
    print(f"  Notes: {args.note_min}-{args.note_max}")
    print(f"  Directory: {args.outdir}/")


if __name__ == '__main__':
    main()
