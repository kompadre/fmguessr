#!/usr/bin/env python3
"""
Train a 1D ResNet audio encoder for DX7 timbre embedding.

Input: directory of WAV files + manifest.csv (from generate_encoder_dataset.py)
Output: encoder.onnx (ONNX model for Go inference)

Architecture:
    Conv1d(1→64, k=80, s=4) → BN → GELU → MaxPool(4)
    Conv1d(64→128, k=3, s=1) → BN → GELU → MaxPool(4)
    Conv1d(128→256, k=3, s=1) → BN → GELU → MaxPool(4)
    Conv1d(256→256, k=3, s=1) → BN → GELU
    AdaptiveAvgPool1d(1) → Linear(256→128) → L2Normalize

Memory-efficient design:
    - WAVs loaded on-the-fly (not pre-loaded into RAM)
    - Windows extracted per-sample in __getitem__
    - Augmentation (pitch shift, velocity scale) applied on-the-fly
    - num_workers=0 to avoid fork-copy memory explosion

Usage:
    python scripts/train_audio_encoder.py --datadir encoder_wavs --output encoder.onnx
    python scripts/train_audio_encoder.py --datadir encoder_wavs --output encoder.onnx --epochs 100 --batch-size 128
"""

import argparse
import csv
import os
import numpy as np
import wave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 44100
WINDOW_SAMPLES = 11025   # 250ms at 44.1kHz
HOP_SAMPLES = 1323       # 30ms at 44.1kHz
RENDER_DUR_S = 3.0


def load_wav(path):
    """Load mono WAV file, return float32 samples normalized to [-1, 1]."""
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        n_samples = len(raw_arr) // 3
        int32 = np.zeros(n_samples, dtype=np.int32)
        for i in range(3):
            int32 |= raw_arr[i::3].astype(np.int32) << (i * 8)
        int32[int32 >= 2**23] -= 2**24
        samples = int32.astype(np.float32) / (2**23 - 1)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples


class EncoderDataset(Dataset):
    """Memory-efficient dataset that loads WAVs on-the-fly.

    Each __getitem__ call:
    1. Loads the WAV file from disk (OS page cache handles caching)
    2. Picks a random 250ms window
    3. Applies on-the-fly augmentation (pitch shift, velocity scale)
    4. Returns (anchor_window, positive_window, patch_id)
    """

    def __init__(self, datadir, augment_pitch=True, augment_velocity=True,
                 pitch_range=(-50, 50), velocity_tiers=(32, 64, 96, 127)):
        self.datadir = datadir
        self.augment_pitch = augment_pitch
        self.augment_velocity = augment_velocity
        self.pitch_range = pitch_range
        self.velocity_tiers = velocity_tiers

        # Read manifest
        manifest_path = os.path.join(datadir, 'manifest.csv')
        self.entries = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({
                    'patch_id': int(row['patch_id']),
                    'midi_note': int(row['midi_note']),
                    'wav_file': row['wav_file'],
                })

        # Group by patch_id for positive pair sampling
        self.patch_to_indices = {}
        for i, e in enumerate(self.entries):
            pid = e['patch_id']
            if pid not in self.patch_to_indices:
                self.patch_to_indices[pid] = []
            self.patch_to_indices[pid].append(i)

        self.valid_patches = [pid for pid, idxs in self.patch_to_indices.items()
                              if len(idxs) >= 2]

        print(f"  Dataset: {len(self.entries)} WAVs, {len(self.valid_patches)} patches with ≥2 samples")

    def __len__(self):
        return len(self.entries)

    def _load_and_window(self, wav_path, augment=True):
        """Load WAV, pick a random window, optionally augment."""
        samples = load_wav(wav_path)

        # Trim to max duration
        max_samples = int(RENDER_DUR_S * SAMPLE_RATE)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        # Pick a random window
        if len(samples) < WINDOW_SAMPLES:
            # Too short — pad with zeros
            window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            window[:len(samples)] = samples
        else:
            max_start = len(samples) - WINDOW_SAMPLES
            start = np.random.randint(0, max(1, max_start))
            window = samples[start:start + WINDOW_SAMPLES].copy()

        if augment:
            # Velocity scaling (random tier)
            if self.augment_velocity and np.random.random() < 0.5:
                vel = np.random.choice(self.velocity_tiers)
                scale_map = {127: 1.0, 96: 0.85, 64: 0.60, 32: 0.35}
                window *= scale_map.get(vel, 1.0)

            # Pitch shift by resampling
            if self.augment_pitch and np.random.random() < 0.5:
                cents = np.random.randint(self.pitch_range[0], self.pitch_range[1] + 1)
                if cents != 0:
                    factor = 2.0 ** (cents / 1200.0)
                    n_out = int(WINDOW_SAMPLES / factor)
                    if n_out >= WINDOW_SAMPLES // 2:
                        indices = np.arange(n_out) * factor
                        indices = np.clip(indices, 0, WINDOW_SAMPLES - 1).astype(int)
                        window = window[indices][:WINDOW_SAMPLES]
                        if len(window) < WINDOW_SAMPLES:
                            pad = np.zeros(WINDOW_SAMPLES - len(window), dtype=np.float32)
                            window = np.concatenate([window, pad])

        # Normalize RMS to 0.1
        rms = np.sqrt(np.mean(window ** 2))
        if rms > 1e-10:
            window *= 0.1 / rms

        return window

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav_path = os.path.join(self.datadir, entry['wav_file'])
        patch_id = entry['patch_id']

        # Anchor: load with augmentation
        anchor = self._load_and_window(wav_path, augment=True)

        # Positive: same patch, different WAV or different window
        pos_indices = self.patch_to_indices[patch_id]
        if len(pos_indices) > 1:
            # Pick a different sample from the same patch
            pos_idx = idx
            for _ in range(10):
                candidate = pos_indices[np.random.randint(len(pos_indices))]
                if candidate != idx:
                    pos_idx = candidate
                    break
            pos_entry = self.entries[pos_idx]
            pos_wav_path = os.path.join(self.datadir, pos_entry['wav_file'])
            positive = self._load_and_window(pos_wav_path, augment=True)
        else:
            # Only one sample for this patch — use a different window from same WAV
            positive = self._load_and_window(wav_path, augment=True)

        return (
            torch.from_numpy(anchor).unsqueeze(0),      # (1, 11025)
            torch.from_numpy(positive).unsqueeze(0),     # (1, 11025)
            torch.tensor(patch_id, dtype=torch.long),
        )


class AudioEncoder(nn.Module):
    """1D ResNet for audio embedding extraction."""

    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 4)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 4)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 4)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        # Manual L2 normalize (F.normalize causes ONNX export issues)
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-12)
        x = x / norm
        return x


def nt_xent_loss(z_anchor, z_positive, temperature=0.1):
    """NT-Xent contrastive loss."""
    batch_size = z_anchor.shape[0]
    z = torch.cat([z_anchor, z_positive], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=sim.device).bool()
    sim.masked_fill_(mask, -1e9)
    pos_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z_anchor.device)
    return F.cross_entropy(sim, pos_indices)


def train(args):
    print(f"Loading dataset from: {args.datadir}")
    dataset = EncoderDataset(
        args.datadir,
        augment_pitch=args.augment_pitch,
        augment_velocity=args.augment_velocity,
    )

    # num_workers=0: main process loads data (avoids fork memory explosion)
    # pin_memory=False: we're on CPU, no GPU copy needed
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioEncoder(emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters, emb_dim={args.emb_dim}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}, Workers: 0 (main process)")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (anchor, positive, patch_id) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)

            z_anchor = model(anchor)
            z_positive = model(positive)

            loss = nt_xent_loss(z_anchor, z_positive, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx+1}: loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output.replace('.onnx', '.pt'))

    # Export to ONNX
    print(f"\nExporting to ONNX: {args.output}")
    model.eval()
    model.load_state_dict(torch.load(args.output.replace('.onnx', '.pt'), weights_only=True, map_location='cpu'))

    dummy = torch.randn(1, 1, WINDOW_SAMPLES)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=['audio'],
        output_names=['embedding'],
        dynamic_axes={
            'audio': {0: 'batch'},
            'embedding': {0: 'batch'},
        },
        opset_version=18,
        dynamo=False,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    test_input = torch.randn(2, 1, WINDOW_SAMPLES)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"  Verified: input {test_input.shape} → output {test_output.shape}")
    print(f"  Norms: {torch.norm(test_output, dim=1).tolist()}")

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nDone! Model size: {size_kb:.1f} KB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 1D ResNet audio encoder')
    parser.add_argument('--datadir', required=True, help='Directory with WAVs + manifest.csv')
    parser.add_argument('--output', default='encoder.onnx', help='Output ONNX model path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--emb-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.1, help='NT-Xent temperature')
    parser.add_argument('--augment-pitch', action='store_true', default=True, help='Pitch shift augmentation')
    parser.add_argument('--no-augment-pitch', dest='augment_pitch', action='store_false')
    parser.add_argument('--augment-velocity', action='store_true', default=True, help='Velocity augmentation')
    parser.add_argument('--no-augment-velocity', dest='augment_velocity', action='store_false')
    args = parser.parse_args()

    train(args)
