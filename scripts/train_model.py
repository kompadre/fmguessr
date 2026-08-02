#!/usr/bin/env python3
"""
Train DX7 Parameter Estimation CNN and export to ONNX.

Usage:
    python scripts/train_model.py [--dataset dataset.npz] [--output model.onnx] [--epochs 100]
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class DX7ParameterEstimator(nn.Module):
    """Lightweight MLP for DX7 timbre parameter estimation.
    
    Predicts only continuous params (fines, levels, feedback).
    Algorithm and coarse ratios are handled by the GA.
    Input: 256-bin timbre spectrum
    Output: 13 continuous values (6 fines + 6 levels + 1 feedback)
    """

    def __init__(self):
        super(DX7ParameterEstimator, self).__init__()

        # Feature extractor (256 -> 128 -> 64)
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # Output: 13 continuous params (6 fines + 6 levels + 1 feedback)
        # Raw logits — sigmoid is applied inside BCEWithLogitsLoss for stable gradients.
        self.head_continuous = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 13),
        )

    def forward(self, x):
        # Input: (Batch, 1, 256) -> squeeze to (Batch, 256)
        feat = x.squeeze(1)
        feat = self.shared(feat)  # -> (Batch, 64)

        continuous_out = self.head_continuous(feat)

        return {
            "continuous": continuous_out
        }


class DX7MaskedMultiTaskLoss(nn.Module):
    """Multi-task loss for DX7 parameter estimation with operator level masking."""

    def __init__(self, w_algo=1.0, w_coarse=2.0, w_cont=5.0, level_threshold=0.05):
        super(DX7MaskedMultiTaskLoss, self).__init__()
        self.w_algo = w_algo
        self.w_coarse = w_coarse
        self.w_cont = w_cont
        self.level_threshold = level_threshold
        
        # reduction='none' allows us to apply per-operator masks before averaging
        self.ce_loss_none = nn.CrossEntropyLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        # 1. Algorithm Loss (Global routing)
        loss_algo = self.ce_loss(predictions["algo_logits"], targets["algo"])

        # 2. Level-Masked Coarse Frequency Loss
        # Extract normalized operator levels (assuming continuous targets indices 1..6 represent Op1..Op6 levels)
        # Shape: (batch_size, 6)
        op_levels = targets["continuous"][:, 1:7]
        
        # Create a soft or hard mask: operators above threshold get full/proportional weight
        # Hard mask: 1.0 if level > threshold else 0.0
        op_mask = (op_levels > self.level_threshold).float()

        total_masked_coarse_loss = 0.0
        total_active_ops = 0.0

        for op in range(6):
            # Compute unreduced CrossEntropy per sample: shape (batch_size,)
            raw_ce = self.ce_loss_none(
                predictions["coarse_logits"][:, op, :], 
                targets["coarse"][:, op]
            )
            
            # Mask out silent operators for this specific operator index
            mask = op_mask[:, op]
            masked_ce = raw_ce * mask
            
            total_masked_coarse_loss += masked_ce.sum()
            total_active_ops += mask.sum()

        # Average loss ONLY across active operators (avoid divide-by-zero with clamp)
        loss_coarse = total_masked_coarse_loss / torch.clamp(total_active_ops, min=1.0)

        # 3. Continuous Loss (Feedback + Operator Levels)
        loss_cont = self.mse_loss(predictions["continuous"], targets["continuous"])

        # Total weighted loss
        total_loss = (self.w_algo * loss_algo) + (self.w_coarse * loss_coarse) + (self.w_cont * loss_cont)

        return total_loss, {
            "loss_algo": loss_algo.item(),
            "loss_coarse": loss_coarse.item(),
            "loss_cont": loss_cont.item(),
            "active_op_ratio": (total_active_ops / (op_levels.size(0) * 6)).item()
        }

class DX7Dataset(Dataset):
    """Dataset for DX7 continuous parameter estimation."""

    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        # Add channel dimension: (256,) -> (1, 256)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()

        # Extract continuous params: 6 fines + 6 levels + 1 feedback = 13
        continuous = np.zeros(13, dtype=np.float32)
        for op in range(6):
            base = 1 + op * 8
            continuous[op] = label[base + 2]      # FreqFine
            continuous[op + 6] = label[base + 3]  # OutputLevel
        continuous[12] = label[49]                # Feedback

        return {
            "spec": spec_tensor,
            "continuous": torch.tensor(continuous, dtype=torch.float32)
        }


class DX7MultiTaskLoss(nn.Module):
    """BCEWithLogitsLoss for continuous timbre parameters (raw logits)."""

    def __init__(self):
        super(DX7MultiTaskLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        # Raw logits vs [0,1] targets — sigmoid applied internally
        loss_cont = self.bce_loss(predictions["continuous"], targets["continuous"])

        return loss_cont, {
            "loss_cont": loss_cont.item()
        }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        specs = batch["spec"].to(device)
        targets = {"continuous": batch["continuous"].to(device)}

        predictions = model(specs)
        loss, _ = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_samples = 0
    mae_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            specs = batch["spec"].to(device)
            targets = {"continuous": batch["continuous"].to(device)}

            predictions = model(specs)
            loss, _ = criterion(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            # MAE metric on sigmoid-activated outputs (meaningful [0,1] range)
            mae_sum += (torch.sigmoid(predictions["continuous"]) - targets["continuous"]).abs().mean().item()
            total_samples += specs.size(0)

    return {
        "loss": total_loss / n_batches,
        "mae": mae_sum / n_batches
    }


def export_to_onnx(model, output_path, device):
    """Export trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 1, 256).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["continuous"],
        dynamo=False,
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "continuous": {0: "batch_size"}
        }
    )
    print(f"Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train DX7 parameter estimation CNN')
    parser.add_argument('--dataset', default='dataset_timbre.npz', help='Input dataset')
    parser.add_argument('--output', default='model.onnx', help='Output ONNX model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)
    # Support both old key ("spectrograms") and new key ("timbre_specs")
    if "timbre_specs" in data:
        spectrograms = data["timbre_specs"]
        print(f"  Loaded timbre specs: {spectrograms.shape}")
    elif "spectrograms" in data:
        spectrograms = data["spectrograms"]
        print(f"  Loaded spectrograms: {spectrograms.shape}")
    else:
        raise KeyError(f"Dataset must contain 'timbre_specs' or 'spectrograms', got keys: {list(data.keys())}")
    labels = data["labels"]
    print(f"  {len(spectrograms)} samples, shape {spectrograms.shape}")

    # Split into train/val
    n_total = len(spectrograms)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = DX7Dataset(spectrograms[train_indices], labels[train_indices])
    val_dataset = DX7Dataset(spectrograms[val_indices], labels[val_indices])

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
    model = DX7ParameterEstimator().to(device)
    if True: 
        criterion = DX7MultiTaskLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)

            scheduler.step(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), "model_best.pth")
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                      f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                      f"mae={val_metrics['mae']:.4f}")

            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and export
    model.load_state_dict(torch.load("model_best.pth"))
    export_to_onnx(model, args.output, device)

    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnxruntime as ort
    import time as _time
    ort_session = ort.InferenceSession(args.output)
    dummy = np.random.randn(1, 1, 256).astype(np.float32)
    outputs = ort_session.run(None, {"spectrogram": dummy})
    print(f"  ONNX output shapes: {[o.shape for o in outputs]}")
    # Quick timing
    times = []
    for _ in range(10):
        t0 = _time.perf_counter()
        ort_session.run(None, {"spectrogram": dummy})
        times.append(_time.perf_counter() - t0)
    avg_ms = sum(times) / len(times) * 1000
    print(f"  Inference time: ~{avg_ms:.1f}ms")

    # Clean up
    os.remove("model_best.pth")
    print(f"Training complete! Model saved to {args.output}")


if __name__ == '__main__':
    main()
