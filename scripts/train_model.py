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
    """4-layer CNN for DX7 parameter estimation from mel spectrograms."""

    def __init__(self):
        super(DX7ParameterEstimator, self).__init__()

        # Convolutional Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Shared Dense Bottleneck
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Head A: Algorithm (32 possible algorithms)
        self.head_algo = nn.Linear(512, 32)

        # Head B: Coarse Ratios (6 operators x 32 possible coarse values)
        self.head_coarse = nn.Linear(512, 6 * 32)

        # Head C: Continuous Parameters (6 Op Levels + 6 Op Fines + 1 Feedback = 13)
        self.head_continuous = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 13),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc_shared(feat)

        algo_logits = self.head_algo(feat)
        coarse_logits = self.head_coarse(feat).view(-1, 6, 32)
        continuous_out = self.head_continuous(feat)

        return {
            "algo_logits": algo_logits,
            "coarse_logits": coarse_logits,
            "continuous": continuous_out
        }


class DX7MultiTaskLoss(nn.Module):
    """Multi-task loss for DX7 parameter estimation."""

    def __init__(self, w_algo=1.0, w_coarse=2.0, w_cont=5.0):
        super(DX7MultiTaskLoss, self).__init__()
        self.w_algo = w_algo
        self.w_coarse = w_coarse
        self.w_cont = w_cont
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        loss_algo = self.ce_loss(predictions["algo_logits"], targets["algo"])

        loss_coarse = 0.0
        for op in range(6):
            loss_coarse += self.ce_loss(predictions["coarse_logits"][:, op, :], targets["coarse"][:, op])

        loss_cont = self.mse_loss(predictions["continuous"], targets["continuous"])

        total_loss = (self.w_algo * loss_algo) + (self.w_coarse * loss_coarse) + (self.w_cont * loss_cont)

        return total_loss, {
            "loss_algo": loss_algo.item(),
            "loss_coarse": loss_coarse.item(),
            "loss_cont": loss_cont.item()
        }


class DX7Dataset(Dataset):
    """Dataset for DX7 parameter estimation."""

    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        # Add channel dimension: (128, 128) -> (1, 128, 128)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()

        # Parse label into targets dict
        algo = int(round(label[0] * 31))  # Denormalize algorithm
        algo = max(0, min(31, algo))

        coarse = np.zeros(6, dtype=np.int64)
        for op in range(6):
            base = 1 + op * 8
            coarse[op] = int(round(label[base + 1] * 31))  # Denormalize coarse
            coarse[op] = max(0, min(31, coarse[op]))

        # Extract continuous params: 6 fines + 6 levels + 1 feedback = 13
        continuous = np.zeros(13, dtype=np.float32)
        for op in range(6):
            base = 1 + op * 8
            continuous[op] = label[base + 2]      # FreqFine (already [0,1])
            continuous[op + 6] = label[base + 3]  # OutputLevel (already [0,1])
        continuous[12] = label[49]  # Feedback (already [0,1])

        return {
            "spec": spec_tensor,
            "algo": torch.tensor(algo, dtype=torch.long),
            "coarse": torch.tensor(coarse, dtype=torch.long),
            "continuous": torch.tensor(continuous, dtype=torch.float32)
        }


class DX7MultiTaskLoss(nn.Module):
    """Multi-task loss for DX7 parameter estimation."""

    def __init__(self, w_algo=1.0, w_coarse=2.0, w_cont=5.0):
        super(DX7MultiTaskLoss, self).__init__()
        self.w_algo = w_algo
        self.w_coarse = w_coarse
        self.w_cont = w_cont
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        loss_algo = self.ce_loss(predictions["algo_logits"], targets["algo"])

        loss_coarse = 0.0
        for op in range(6):
            loss_coarse += self.ce_loss(predictions["coarse_logits"][:, op, :], targets["coarse"][:, op])

        loss_cont = self.mse_loss(predictions["continuous"], targets["continuous"])

        total_loss = (self.w_algo * loss_algo) + (self.w_coarse * loss_coarse) + (self.w_cont * loss_cont)

        return total_loss, {
            "loss_algo": loss_algo.item(),
            "loss_coarse": loss_coarse.item(),
            "loss_cont": loss_cont.item()
        }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        specs = batch["spec"].to(device)
        targets = {
            "algo": batch["algo"].to(device),
            "coarse": batch["coarse"].to(device),
            "continuous": batch["continuous"].to(device)
        }

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
    algo_correct = 0
    coarse_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            specs = batch["spec"].to(device)
            targets = {
                "algo": batch["algo"].to(device),
                "coarse": batch["coarse"].to(device),
                "continuous": batch["continuous"].to(device)
            }

            predictions = model(specs)
            loss, _ = criterion(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            # Accuracy metrics
            algo_pred = predictions["algo_logits"].argmax(dim=1)
            algo_correct += (algo_pred == targets["algo"]).sum().item()

            coarse_pred = predictions["coarse_logits"].argmax(dim=2)
            coarse_correct += (coarse_pred == targets["coarse"]).all(dim=1).sum().item()

            total_samples += specs.size(0)

    return {
        "loss": total_loss / n_batches,
        "algo_acc": algo_correct / total_samples,
        "coarse_acc": coarse_correct / total_samples
    }


def export_to_onnx(model, output_path, device):
    """Export trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["algo_logits", "coarse_logits", "continuous"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "algo_logits": {0: "batch_size"},
            "coarse_logits": {0: "batch_size"},
            "continuous": {0: "batch_size"}
        }
    )
    print(f"Model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train DX7 parameter estimation CNN')
    parser.add_argument('--dataset', default='dataset.npz', help='Input dataset')
    parser.add_argument('--output', default='model.onnx', help='Output ONNX model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)
    spectrograms = data["spectrograms"]
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
    criterion = DX7MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
                  f"algo_acc={val_metrics['algo_acc']:.3f} coarse_acc={val_metrics['coarse_acc']:.3f}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and export
    model.load_state_dict(torch.load("model_best.pth"))
    export_to_onnx(model, args.output, device)

    # Verify ONNX model
    print("Verifying ONNX model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession(args.output)
    dummy = np.random.randn(1, 1, 128, 128).astype(np.float32)
    outputs = ort_session.run(None, {"spectrogram": dummy})
    print(f"  ONNX output shapes: {[o.shape for o in outputs]}")
    print(f"  Inference time: ~{ort_session.get_profiling_result().average_time * 1000:.1f}ms")

    # Clean up
    os.remove("model_best.pth")
    print(f"Training complete! Model saved to {args.output}")


if __name__ == '__main__':
    main()
