# fmguessr

`fmguessr` is a 6-operator FM synthesizer patch generator and matcher. It uses continuous MLP neural network prediction, 256-bin sustained timbre STFT spectra, a 32-island genetic algorithm, and live multi-threaded hill-climbing optimization to match target audio WAV files and generate DX7-compatible System Exclusive (`.syx`) bank files.

## Features

- **256-Bin Timbre STFT Spectrum:** Sustained-window timbre extraction (skips 80ms attack transient, identifies stable 300ms window via minimum spectral flux, applies temporal median STFT) compressed into 256 log-frequency bins.
- **Preset Codebook:** Fast K-Nearest Neighbors (KNN) lookup to seed optimization with known DX7 presets via timbre distance, bypassing blind parameter guessing.
- **MLP Predictor (~2ms):** PyTorch MLP model predicting continuous DX7 parameters (fines, levels, feedback) directly from the 256-bin timbre spectrum.
- **Parallel Live Optimizer:** Multi-threaded coordinate descent hill-climber (`~400 candidates` evaluated in parallel across all CPU cores in `~0.1–0.3s`).
- **Acceleration Trough Cacher:** Calculates discrete 2nd-derivative acceleration scores ($d_{i-1} - 2d_i + d_{i+1}$) on parameter sweeps to store sweet-spot troughs per operator and execute multi-step leaps over distance valleys.
- **Topological Algorithm Jumps:** Tests all 32 operator routing topologies in a single parallel batch pass to escape local minima.
- **Hybrid Island Seeding:** Seeds 32 algorithm islands with priority NN/KNN matches, combinatorial acceleration trough permutations, and a random populace for exploration diversity.
- **Template Preservation:** Maintains non-tuned DX7 parameters (LFO, Pitch Envelopes) from matched presets for high-fidelity reconstruction.
- **Early Termination:** Stops immediately when an exact match (distance < 10⁻⁶) is found.
- **Stagnation Recovery:** Detects plateaus and injects fresh codebook mutants with boosted mutation rate.

## Installation

### Prerequisites

- Go 1.20+
- C++ compiler (for MSFA engine)
- Python 3.8+ with PyTorch (only for training the NN model)

### Build

```bash
make
```

## Testing

```bash
make test
```

## Usage

### 1. Build a Patch Codebook

Index your `.syx` collection to create a searchable timbre database:

```bash
./index /path/to/syx_library/
# or index specific files
./index Dexed_01.syx SynprezFM_01.syx
```

### 2. Match a Target Audio File

Find the best DX7 parameters to replicate a WAV file (the best match is automatically indexed into the codebook by default):

```bash
# Standard KNN + GA + LiveOptimizer matching
GOMAXPROCS=16 ./match -v target.wav -syx result.syx

# With NN predictor (faster, requires trained model)
./match -model model.onnx -v target.wav -syx result.syx

# Disable automatic codebook indexing
./match -auto-index=false -v target.wav
```

### 3. Play or Export the Match

Render the generated patch to verify the sound:

```bash
./play result.syx 0 C3 -o match.wav
```

### 4. Run the REST API Server

Start the REST API server for frontend interaction:

```bash
./server -port 8080 -model smaller_model.onnx -codebook dx7_codebook.bin
```

- **`POST /api/probe`**: Upload WAV file to instantly analyze envelope ($A, D, S, R$), gate duration ($T_{\text{gate}}$), note-on/note-off positions, pitch heuristics, and check if an exact match exists in the codebook (`seen_before: true` for distance $\le 0.001$). Supports form overrides (`note` e.g. `"C3"`, `attack_ms`, `decay_ms`, `sustain_level`, `release_ms`, `gate_ms`, `brightness`, `harmonicity`).
- **`POST /api/match`**: Upload WAV file (`multipart/form-data`, `< 800 KB` limit, truncated to 4.0s max). Returns HTTP 202 Accepted with job ID. Supports feature overrides.
- **`GET /api/status/:id`**: Poll job status and real-time progress (0–100%). Returns completed JSON result with top 5 match WAV URLs and `.syx` bank URL.
- **`GET /exports/*`**: Static file server for rendered candidate WAV files and `.syx` bank files (with CORS enabled).

### 5. Train the Neural Network (Optional)

```bash
# Generate training dataset from codebook
python scripts/generate_dataset.py --codebook dx7_codebook.bin

# Train MLP model and export to ONNX
python scripts/train_model.py --dataset dataset.npz --output model.onnx
```

## Architecture

```
target.wav
  ├─→ Extract 256-Bin Timbre STFT Spectrum (300ms sustained window, min flux, temporal median)
  ├─→ KNN Timbre Codebook Search
  ├─→ [Optional] MLP Predictor (~2ms)
  │     └─→ Continuous prediction (fine frequencies, levels, feedback)
  ├─→ LiveOptimizer Acceleration Sweep (Piano 1 / NN base)
  │     └─→ Discovers top discrete 2nd-derivative acceleration troughs per knob
  └─→ Full 32-Island Niche GA Pipeline
        └─→ Seeding: Priority NN/KNN + Acceleration Trough Permutations + Random Populace
        └─→ Phase 1: Niche island evolution (20 gens across 32 algorithms)
        └─→ Phase 2: Global convergence with stagnation recovery
        └─→ Phase 3: Parallel LiveOptimizer (acceleration leaps & topological algorithm jumps)
        └─→ Early termination at distance < 10⁻⁶
```

## License

MIT License. See `LICENSE` for details.
