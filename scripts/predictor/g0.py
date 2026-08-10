#!/usr/bin/env python3
"""
Ground-truth diagnostic: verify loss→0 and overfit convergence
for the DX7 continuous-parameter MLP (DX7ParameterEstimator).

Usage:
    python scripts/g0.py [--dataset dataset.npz]
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from train_model import DX7Dataset, DX7MultiTaskLoss, DX7ParameterEstimator


def run_ground_truth_test():
    print("=== Running Ground-Truth Pipeline Diagnostic ===")

    parser = argparse.ArgumentParser(description="Ground-truth diagnostic for DX7 MLP")
    parser.add_argument("--dataset", default="dataset_timbre.npz", help="Input dataset (.npz)")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)
    if "timbre_specs" in data:
        spectrograms = data["timbre_specs"]
        print(f"  Loaded timbre specs: {spectrograms.shape}")
    elif "spectrograms" in data:
        spectrograms = data["spectrograms"]
        if spectrograms.ndim == 3:
            N = spectrograms.shape[0]
            spectrograms = spectrograms.reshape(N, -1)
            print(f"  Reshaped mel spectrograms to: {spectrograms.shape}")
        else:
            print(f"  Loaded spectrograms: {spectrograms.shape}")
    else:
        raise KeyError(f"Need 'timbre_specs' or 'spectrograms', got: {list(data.keys())}")
    labels = data["labels"]
    print(f"  {len(spectrograms)} samples, shape {spectrograms.shape}")

    # Split into train/val
    n_total = len(spectrograms)
    n_train = int(n_total * 0.8)

    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]

    dataset = DX7Dataset(spectrograms[train_indices], labels[train_indices])

    # Select up to 16 samples
    n_samples = min(16, len(dataset))
    if n_samples < 16:
        print(f"  Warning: only {n_samples} samples available")
    stride = max(1, len(dataset) // n_samples)
    diverse_indices = [i * stride for i in range(n_samples)]
    diverse_indices = [i for i in diverse_indices if i < len(dataset)]

    diverse_subset = torch.utils.data.Subset(dataset, diverse_indices)
    loader = DataLoader(diverse_subset, batch_size=n_samples, shuffle=False)

    batch = next(iter(loader))
    specs = batch["spec"]
    # DataLoader already provides (B, 1, 256) — no unsqueeze needed

    targets = {"continuous": batch["continuous"]}

    print(f"  Continuous range: min={targets['continuous'].min().item():.3f}  max={targets['continuous'].max().item():.3f}")

    criterion = DX7MultiTaskLoss()

    # -------------------------------------------------------------------------
    # TEST 1: Ground-Truth Loss = Theoretical Minimum
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Ground-Truth Loss Evaluation ---")
    # BCEWithLogitsLoss minimum is the binary entropy: -t*log(t) - (1-t)*log(1-t)
    # At the optimal logit x = logit(t), loss = binary_entropy(t), NOT zero.
    def logit(t):
        t = t.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(t / (1.0 - t))

    def binary_entropy(t):
        t = t.clamp(1e-7, 1.0 - 1e-7)
        return -(t * torch.log(t) + (1.0 - t) * torch.log(1.0 - t))

    perfect_logits = {"continuous": logit(targets["continuous"])}

    gt_loss, details = criterion(perfect_logits, targets)
    expected_min = binary_entropy(targets["continuous"]).mean().item()

    print(f"Total Loss:       {gt_loss.item():.8f}")
    print(f"Expected minimum: {expected_min:.8f}")
    print(f"  └─ Cont BCE:    {details['loss_cont']:.8f}")

    assert abs(gt_loss.item() - expected_min) < 1e-4, \
        f"Loss {gt_loss.item():.6f} should match theoretical minimum {expected_min:.6f}"
    print("PASS: Loss matches theoretical minimum (optimal logits produce minimum BCE).")

    # -------------------------------------------------------------------------
    # TEST 2: Overfit Sprint (200 epochs on 16 samples)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: 200-Epoch Overfit Test ---")
    param_names = ['FreqFine1','FreqFine2','FreqFine3','FreqFine4','FreqFine5','FreqFine6',
                   'OutLevel1','OutLevel2','OutLevel3','OutLevel4','OutLevel5','OutLevel6','Feedback']

    model = DX7ParameterEstimator()
    # Head gets 2× LR (matching train_model.py)
    optimizer = torch.optim.Adam([
        {'params': model.shared.parameters(), 'lr': 1e-3, 'weight_decay': 0},
        {'params': model.head_continuous.parameters(), 'lr': 2e-3, 'weight_decay': 0},
    ], lr=1e-3, weight_decay=0)

    model.train()
    for epoch in range(1, 2001):
        optimizer.zero_grad()
        # Use eval mode to disable dropout — this is a pure memorization test
        model.eval()
        preds = model(specs)
        loss, details = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        if epoch == 1:
            initial_loss = loss.item()
        if epoch % 500 == 0 or epoch == 1:
            print(f"  Epoch {epoch:04d} | Loss: {loss.item():.5f}")

    # Evaluate MAE on sigmoid-activated outputs (DX7 steps)
    final_loss = loss.item()
    model.eval()
    with torch.no_grad():
        final_logits = model(specs)["continuous"]
        final_probs = torch.sigmoid(final_logits)
        per_param_mae = (final_probs - targets["continuous"]).abs().mean(dim=0)
        overall_mae = per_param_mae.mean().item()

    print(f"\nFinal BCE loss: {final_loss:.6f}  (theoretical min: ~{expected_min:.4f})")
    print(f"Overall MAE: {overall_mae:.6f} DX7 steps")
    print("Per-param MAE:")
    for name, mae in zip(param_names, per_param_mae):
        print(f"  {name:12s}: {mae:.6f}")

    if overall_mae < 0.005:
        print(f"PASS: MAE {overall_mae:.5f} < 0.005 (half DX7 step).")
    else:
        print(f"WARNING: MAE {overall_mae:.5f} >= 0.005.")

    # -------------------------------------------------------------------------
    # TEST 3: Single-Sample Round-Trip
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Trained Model Round-Trip ---")
    single_spec = specs[:1]
    single_target = {"continuous": targets["continuous"][:1]}

    model2 = DX7ParameterEstimator()
    opt2 = torch.optim.Adam(model2.parameters(), lr=5e-4)
    model2.train()
    for epoch in range(1, 501):
        opt2.zero_grad()
        pred = model2(single_spec)
        loss, _ = criterion(pred, single_target)
        loss.backward()
        opt2.step()

    model2.eval()
    with torch.no_grad():
        pred = model2(single_spec)
        # Raw logits → sigmoid for comparison with [0,1] targets
        final_cont = torch.sigmoid(pred["continuous"])
        gt_cont = single_target["continuous"]
        mse = torch.nn.functional.mse_loss(final_cont, gt_cont).item()
        max_err = (final_cont - gt_cont).abs().max().item()

    print(f"  Single-sample MSE:     {mse:.10f}")
    print(f"  Single-sample max err: {max_err:.10f}")

    if mse < 1e-4:
        print("PASS: Trained model reproduces ground truth to < 1e-4 MSE.")
    else:
        print(f"WARNING: MSE {mse:.2e} exceeds 1e-4 threshold.")

    print("\n=== Diagnostic Complete ===")


if __name__ == "__main__":
    run_ground_truth_test()
