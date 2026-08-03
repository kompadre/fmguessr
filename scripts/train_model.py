#!/usr/bin/env python3
"""
Train DX7 Multi-Head Parameter Estimator and export to ONNX.

Usage:
    python scripts/train_model.py [--dataset dataset.npz] [--output model_probe.onnx] [--epochs 100]
    python scripts/train_model.py --dataset dataset.npz --output model_probe.onnx --dx7-weight 2.0

Multi-head architecture:
    Head 1 (probes): 9 acoustic parameters (ADSR, brightness, harmonicity, pitch)
    Head 2 (dx7):    53 normalized DX7 byte parameters (direct prediction)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROBE_LABELS = 9
DX7_LABELS = 53   # Algorithm(1) + 6 ops × 8 params(48) + Feedback(1) + PitchEnv(3)
TIMBRE_BINS = 256
INPUT_DIM = TIMBRE_BINS + 1  # 256 spectrum + 1 MIDI note


class ProbeEstimator(nn.Module):
    """Multi-head MLP for predicting probe + DX7 parameters from timbre spectrum.

    Input:  (Batch, 1, 257) — 256 timbre bins + 1 MIDI note (normalized)
    Output: dict with "probe" (Batch, 9) and optionally "dx7" (Batch, 53)
    """

    def __init__(self, use_dx7_head=False):
        super(ProbeEstimator, self).__init__()
        self.use_dx7_head = use_dx7_head

        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.probe_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, PROBE_LABELS),
        )

        if use_dx7_head:
            self.dx7_head = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, DX7_LABELS),
            )

    def forward(self, x):
        feat = x.squeeze(1)  # (Batch, 257)
        feat = self.shared(feat)  # -> (Batch, 64)
        out = {"probe": self.probe_head(feat)}
        if self.use_dx7_head:
            out["dx7"] = self.dx7_head(feat)
        return out


class ProbeDataset(Dataset):
    """Dataset for multi-head probe + DX7 parameter estimation."""

    def __init__(self, timbre_specs, probe_labels, dx7_labels=None):
        self.timbre_specs = timbre_specs
        self.probe_labels = probe_labels
        self.dx7_labels = dx7_labels

    def __len__(self):
        return len(self.timbre_specs)

    def __getitem__(self, idx):
        spec = self.timbre_specs[idx]  # (256,)
        probe = self.probe_labels[idx]  # (9,)

        # Reconstruct MIDI note from semitone + octave for input
        semitone = int(round(probe[7] * 11.0))  # denormalize
        octave = int(round(probe[8] * 8.0))      # denormalize
        midi_note = semitone + (octave + 1) * 12
        midi_note = max(0, min(127, midi_note))
        midi_norm = float(midi_note) / 127.0

        input_vec = np.concatenate([spec, [midi_norm]]).astype(np.float32)  # (257,)

        # Add channel dimension: (257,) -> (1, 257)
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).float()

        item = {
            "input": input_tensor,
            "probe": torch.tensor(probe, dtype=torch.float32),
        }
        if self.dx7_labels is not None:
            item["dx7"] = torch.tensor(self.dx7_labels[idx], dtype=torch.float32)
        return item


def compute_loss(model, batch, probe_criterion, dx7_criterion, dx7_weight, device):
    """Compute combined multi-head loss."""
    inputs = batch["input"].to(device)
    predictions = model(inputs)

    probe_loss = probe_criterion(predictions["probe"], batch["probe"].to(device))

    total_loss = probe_loss
    dx7_loss_val = 0.0
    if "dx7" in predictions and "dx7" in batch:
        dx7_loss = dx7_criterion(predictions["dx7"], batch["dx7"].to(device))
        dx7_loss_val = dx7_loss.item()
        total_loss = total_loss + dx7_weight * dx7_loss

    return total_loss, probe_loss.item(), dx7_loss_val


def train_one_epoch(model, dataloader, probe_criterion, dx7_criterion, dx7_weight, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        loss, _, _ = compute_loss(model, batch, probe_criterion, dx7_criterion, dx7_weight, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, probe_criterion, dx7_criterion, dx7_weight, device):
    model.eval()
    total_loss = 0.0
    probe_mae_sum = 0.0
    dx7_mae_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, _, dx7_loss_val = compute_loss(model, batch, probe_criterion, dx7_criterion, dx7_weight, device)

            total_loss += loss.item()
            n_batches += 1

            # MAE on sigmoid-activated probe outputs
            inputs = batch["input"].to(device)
            predictions = model(inputs)
            probe_mae_sum += (torch.sigmoid(predictions["probe"]) - batch["probe"].to(device)).abs().mean().item()

            # MAE on sigmoid-activated DX7 outputs
            if "dx7" in predictions and "dx7" in batch:
                dx7_mae_sum += (torch.sigmoid(predictions["dx7"]) - batch["dx7"].to(device)).abs().mean().item()

    metrics = {
        "loss": total_loss / n_batches,
        "probe_mae": probe_mae_sum / n_batches,
    }
    if n_batches > 0 and dx7_mae_sum > 0:
        metrics["dx7_mae"] = dx7_mae_sum / n_batches
    return metrics


def export_to_onnx(model, output_path, device, use_dx7_head):
    """Export trained model to ONNX format with dual outputs."""
    model.eval()
    dummy_input = torch.randn(1, 1, INPUT_DIM).to(device)

    output_names = ["probe"]
    dynamic_axes = {
        "spectrogram": {0: "batch_size"},
        "probe": {0: "batch_size"},
    }
    if use_dx7_head:
        output_names.append("dx7")
        dynamic_axes["dx7"] = {0: "batch_size"}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=output_names,
        dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    print(f"Model exported to {output_path} (outputs: {output_names})")


def main():
    parser = argparse.ArgumentParser(description='Train DX7 multi-head parameter estimator')
    parser.add_argument('--dataset', default='dataset_probe.npz', help='Input dataset')
    parser.add_argument('--output', default='model_probe.onnx', help='Output ONNX model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--dx7-weight', type=float, default=2.0, help='Weight for DX7 head loss (0=disable)')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    use_dx7_head = args.dx7_weight > 0

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)

    if "timbre_specs" in data:
        timbre_specs = data["timbre_specs"]
    elif "spectrograms" in data:
        timbre_specs = data["spectrograms"]
    else:
        raise KeyError(f"Dataset must contain 'timbre_specs' or 'spectrograms', got keys: {list(data.keys())}")

    if "probe_labels" in data:
        probe_labels = data["probe_labels"]
    elif "labels" in data:
        labels = data["labels"]
        if labels.shape[1] >= 50:
            print("  Warning: legacy DX7 labels detected. Cannot extract probe features from DX7 params.")
            print("  Please re-run generate_dataset.py with the new probe extraction.")
            sys.exit(1)
        probe_labels = labels[:, :8]
    else:
        raise KeyError(f"Dataset must contain 'probe_labels', got keys: {list(data.keys())}")

    # Load DX7 labels if available
    dx7_labels = None
    if use_dx7_head and "dx7_labels" in data:
        dx7_labels = data["dx7_labels"]
        print(f"  DX7 labels: {dx7_labels.shape}")
    elif use_dx7_head:
        print("  Warning: dx7_labels not found in dataset. Disabling DX7 head.")
        use_dx7_head = False
        args.dx7_weight = 0

    print(f"  {len(timbre_specs)} samples, timbre shape {timbre_specs.shape}, probe shape {probe_labels.shape}")

    # Split into train/val
    n_total = len(timbre_specs)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = ProbeDataset(timbre_specs[train_indices], probe_labels[train_indices],
                                  dx7_labels[train_indices] if dx7_labels is not None else None)
    val_dataset = ProbeDataset(timbre_specs[val_indices], probe_labels[val_indices],
                                dx7_labels[val_indices] if dx7_labels is not None else None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"  Train: {n_train}, Val: {n_val}")

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")

    # Create model
    model = ProbeEstimator(use_dx7_head=use_dx7_head).to(device)
    probe_criterion = nn.BCEWithLogitsLoss()
    dx7_criterion = nn.BCEWithLogitsLoss()

    # Optimizer with separate LR for heads
    head_lr = args.lr * 2
    param_groups = [
        {'params': model.shared.parameters(), 'lr': args.lr, 'weight_decay': 0},
        {'params': model.probe_head.parameters(), 'lr': head_lr, 'weight_decay': 0},
    ]
    if use_dx7_head:
        param_groups.append({'params': model.dx7_head.parameters(), 'lr': head_lr, 'weight_decay': 0})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    if use_dx7_head:
        print(f"  DX7 head weight: {args.dx7_weight}")
    print(f"  Head LR: {head_lr} (shared: {args.lr})")

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, probe_criterion, dx7_criterion,
                                      args.dx7_weight, optimizer, device)
        val_metrics = validate(model, val_loader, probe_criterion, dx7_criterion,
                                args.dx7_weight, device)

        # Use probe MAE for early stopping (primary task)
        val_mae = val_metrics["probe_mae"]
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), "model_probe_best.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            dx7_str = ""
            if "dx7_mae" in val_metrics:
                dx7_str = f" dx7_mae={val_metrics['dx7_mae']:.4f}"
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"probe_mae={val_metrics['probe_mae']:.4f}{dx7_str}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and export
    model.load_state_dict(torch.load("model_probe_best.pth"))
    export_to_onnx(model, args.output, device, use_dx7_head)

    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnxruntime as ort
    import time as _time
    ort_session = ort.InferenceSession(args.output)
    dummy = np.random.randn(1, 1, INPUT_DIM).astype(np.float32)
    outputs = ort_session.run(None, {"spectrogram": dummy})
    print(f"  ONNX output names: {[o.name for o in ort_session.get_outputs()]}")
    print(f"  ONNX output shapes: {[o.shape for o in outputs]}")

    # Quick timing
    times = []
    for _ in range(10):
        t0 = _time.perf_counter()
        ort_session.run(None, {"spectrogram": dummy})
        times.append(_time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    print(f"  Inference time: ~{avg_ms:.1f}ms")

    # Print output layout
    probe_names = ['AttackMs', 'DecayMs', 'SustainLevel', 'ReleaseMs', 'GateMs',
                   'Brightness', 'Harmonicity', 'Semitone', 'Octave']
    print(f"\n  Probe head ({PROBE_LABELS} params):")
    for i, name in enumerate(probe_names):
        print(f"    [{i}] {name}")

    if use_dx7_head:
        dx7_names = (['Algorithm'] +
                     [f'Op{6-i}.Mode' for i in range(6)] +
                     [f'Op{6-i}.Coarse' for i in range(6)] +
                     [f'Op{6-i}.Fine' for i in range(6)] +
                     [f'Op{6-i}.Level' for i in range(6)] +
                     [f'Op{6-i}.Rate1' for i in range(6)] +
                     [f'Op{6-i}.Rate2' for i in range(6)] +
                     [f'Op{6-i}.L3' for i in range(6)] +
                     [f'Op{6-i}.Rate4' for i in range(6)] +
                     ['Feedback', 'Pitch.Rate1', 'Pitch.Lvl1', 'Pitch.Lvl2'])
        print(f"\n  DX7 head ({DX7_LABELS} params):")
        for i, name in enumerate(dx7_names):
            print(f"    [{i:2d}] {name}")

    # Clean up
    os.remove("model_probe_best.pth")
    print(f"\nTraining complete! Model saved to {args.output}")


if __name__ == '__main__':
    main()
