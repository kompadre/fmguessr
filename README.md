# fmguessr

DX7 FM synthesizer patch matcher. Uses genetic algorithms, formant analysis, CNN classification, MLP prediction, and hill-climbing to match target audio WAV files and generate DX7 `.syx` bank files.

**Browser frontend:** [OpenPatches](https://github.com/kompadre/openpatches) (live at [openpatch.es](https://openpatch.es))

## Features

- **Formant Candidate Generation:** Extracts spectral envelope peaks (F1/F2/F3) from target, generates DX7 candidates whose FM sidebands match. Zero-render, pure math.
- **Formant-Constrained Refinement:** Optimizes candidates on formant distance while keeping mel distance within 10% tolerance.
- **CNN Patch Classifier:** 128×128 mel spectrogram → CNN → identifies presets from codebook.
- **MLP Predictor (~2ms):** Predicts DX7 parameters from 256-bin timbre spectrum.
- **Preset Codebook:** KNN lookup via timbre distance for instant GA seeding.
- **32-Island GA:** Niche evolution across algorithm topologies → global convergence → settle phase.
- **LiveOptimizer:** Multi-threaded hill-climbing with acceleration trough cacher.
- **DiffOptimizer:** Jacobian steering on temporal timbre features.
- **Trajectory Distance (dormant):** 1D ResNet encoder for temporal perception. Infrastructure exists but auto-disables due to scale mismatch with mel.
- **Template Preservation:** LFO, pitch envelopes, and scaling from matched presets preserved.

## Build

```bash
make          # Build all tools
make test     # Run all tests
```

Requires Go 1.21+, C++ compiler (MSFA engine), Python 3.8+ with PyTorch (for training).

## Testing

```bash
make test               # Unit + E2E tests
make test-smoke         # Fast e2e smoke tests (no external fixtures)
make test-regression    # Regression tests (-tags regression)
go test -v ./pkg/match/...           # Unit tests only
go test -v -tags regression ./tests/...  # Regression tests only
go test -race ./pkg/match/...        # Race detector
```

## Usage

Flags must come before the WAV path.

### Match a Target

```bash
# Standard GA matching
./match -v target.wav -syx result.syx

# With NN predictor
./match -model model.onnx -v target.wav -syx result.syx

# With trajectory distance (neural encoder)
./match -encoder encoder.onnx -v target.wav -syx result.syx

# Save/load trajectory session (skip recomputation)
./match -encoder encoder.onnx -session-out session.bin -v target.wav
./match -session-in session.bin -v target.wav -syx result.syx

# Classifier-only (fast, no GA)
./match --classifier patch_classifier.onnx --classifier-only -v target.wav -syx result.syx
```

### Play/Export

```bash
./play result.syx 0 C3 -o match.wav
```

### Train the MLP Predictor

```bash
python scripts/predictor/generate_dataset.py --codebook dx7_codebook.bin
python scripts/predictor/train_model.py --dataset scripts/predictor/dataset.npz --output model.onnx
```

### Train the Patch Classifier

```bash
./classifier --codebook dx7_codebook.bin --output dataset_cls.bin --dur 3s
python scripts/classifier/train_classifier.py --dataset dataset_cls.bin --output patch_classifier.onnx
```

### Train the Audio Encoder (trajectory distance)

```bash
python scripts/encoder/generate_encoder_dataset.py --codebook dx7_codebook.bin --outdir encoder_wavs
python scripts/encoder/train_audio_encoder.py --datadir encoder_wavs --output encoder.onnx --epochs 100
```

### Train All Models

```bash
make -f scripts/Makefile all
```

### Index a Codebook

```bash
./index Dexed_01.syx
```

### REST API

```bash
./server -port 8080 -model model.onnx -codebook dx7_codebook.bin
# With trajectory distance:
./server -port 8080 -model model.onnx -codebook dx7_codebook.bin -encoder encoder.onnx
```

- `POST /api/probe` — Analyze WAV envelope/pitch/formants instantly
- `POST /api/match` — Async matching (returns job ID, poll `GET /api/status/:id`)
- `POST /api/play` — Render patch to WAV
- `POST /api/voice` — Extract voice data from .syx
- `POST /api/morph` — Morph two patches with spectral optimization
- `POST /api/export-syx` — Export patches as .syx bank
- `GET /api/health` — Server health check
- `GET /exports/*` — Static files with CORS

See [API.md in openpatches](https://github.com/kompadre/openpatches/blob/main/API.md) for full request/response schemas.

## Architecture

```
target.wav
  ├─→ Feature Extraction (pitch, envelope, brightness, harmonicity)
  ├─→ Formant Extraction (F1/F2/F3 from spectral envelope)
  ├─→ Formant Candidate Generation (map F1/F2/F3 → DX7 operator ratios)
  ├─→ Dual Codebook Search (fingerprint KNN + timbre KNN)
  ├─→ [Optional] MLP Predictor (timbre → DX7 params)
  ├─→ [Optional] CNN Classifier (mel spectrogram → patch prediction)
  ├─→ 32-Island Niche GA
  │     ├─ Phase 1: Niche evolution (algorithm-locked islands)
  │     ├─ Phase 2: Global convergence (cross-island migration)
  │     ├─ Settle: 5 quiet generations to confirm convergence
  │     └─ Phase 3: LiveOptimizer + DiffOptimizer refinement
  ├─→ Formant-Constrained Refinement (minimize formant dist, mel ≤ 110%)
  └─→ DX7 .syx export (top-32 candidates from diverse sources)
```

## Result Buckets

The matcher produces 5 result categories:

- **Top 32** — Diverse candidates from island bests + formant + NN + GA (for .syx export)
- **TimbralTop5** — Best by static timbre distance
- **FullTop5** — Best by temporal distance (DiffOptimizer-refined)
- **FormantTop5** — Best by formant-constrained refinement
- **Favorites** — Candidates appearing in both TimbralTop5 and FullTop5

## License

MIT
