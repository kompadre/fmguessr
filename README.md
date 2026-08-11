# fmguessr

DX7 FM synthesizer patch matcher. Uses genetic algorithms, formant analysis, CNN classification, MLP prediction, and hill-climbing to match target audio WAV files and generate DX7 `.syx` bank files.

**Browser frontend:** [OpenPatches](https://github.com/kompadre/openpatches) (live at [openpatch.es](https://openpatch.es))

## Features

- **Matcher Pipeline:** Modular 10-step pipeline — each step is a `Matcher` interface with early-return support, reconciled into a shared `ResultsCohort` after each step.
- **Formant Candidate Generation:** Extracts spectral envelope peaks (F1/F2/F3) from target, generates DX7 candidates whose FM sidebands match. Zero-render, pure math.
- **Formant-Constrained Refinement:** Optimizes candidates on formant distance while keeping mel distance within 10% tolerance.
- **CNN Patch Classifier:** 128×128 mel spectrogram → CNN → identifies presets from codebook. Can short-circuit the pipeline on high confidence (≥0.99).
- **MLP Predictor (~2ms):** Predicts DX7 parameters from 256-bin timbre spectrum.
- **Preset Codebook:** Dual KNN lookup (fingerprint + timbre) for instant GA seeding.
- **32-Island GA:** Niche evolution across algorithm topologies → global convergence → settle phase. Seeded from codebook, NN, beam search, and LiveOptimizer results.
- **LiveOptimizer:** Multi-threaded hill-climbing with acceleration trough cacher.
- **DiffOptimizer:** Jacobian steering on temporal timbre features.
- **Beam Search:** Constructive operator-by-operator patch building.
- **CSPOA:** Codebook-seeded progressive operator attachment.
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

### Train Models

Training uses `scripts/Makefile` with configurable Python:

```bash
# Train all models
make -f scripts/Makefile all

# Override Python interpreter
make -f scripts/Makefile PYTHON=/path/to/python3.11 predictor

# Individual pipelines
make -f scripts/Makefile predictor       # MLP predictor (generate + train + export)
make -f scripts/Makefile encoder         # Audio encoder (render + train + export)
make -f scripts/Makefile classifier      # Patch classifier (render + train + export)
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
- `GET /exports/*` — Static files (server-generated only, no user uploads stored)

See [API.md in openpatches](https://github.com/kompadre/openpatches/blob/main/API.md) for full request/response schemas.

## Architecture

```
target.wav
  ├─→ Matcher Pipeline (10 steps, each reconciled into ResultsCohort):
  │     1. CodebookSearch    — dual KNN (fingerprint + timbre)
  │     2. FormantGenerate   — spectral envelope → DX7 candidates
  │     3. NNPredict         — MLP probe predictor
  │     4. NNRefine          — LiveOptimizer + DiffOptimizer on best NN
  │     5. BeamSearch        — constructive operator-by-operator
  │     6. CSPOAStep         — progressive operator attachment
  │     7. ClassifierStep    — ONNX CNN (can early-return on conf ≥ 0.99)
  │     8. GAStep            — 32-island niche → global convergence
  │     9. HillClimbStep     — LiveOptimizer + DiffOptimizer on GA best
  │    10. FormantRefineStep — formant-constrained refinement
  ├─→ Finalize: rescue → top-32 assembly → carrier calibration → temporal re-eval
  └─→ DX7 .syx export (top-32 candidates from diverse sources)
```

## Result Buckets

The matcher produces 5 result categories:

- **Top 32** — Diverse candidates from island bests + formant + NN + GA (for .syx export)
- **TimbralTop5** — Best by static timbre distance
- **FullTop5** — Best by temporal distance (DiffOptimizer-refined)
- **FormantTop5** — Best by formant-constrained refinement
- **Favorites** — Candidates appearing in both TimbralTop5 and FullTop5

## Key Types

- `Candidate` — DX7 patch with `RawBase [128]byte`, `Scores [NumScoreTypes]float64`, `Fitness`
- `ResultsCohort` — Best + Timbral[5] + Full[5] + Formant[5] (fixed-size arrays)
- `CandidatePool` — 1024-slot pre-allocated pool (zero-allocation)
- `Matcher` interface — `Name() string` + `Run(ctx, MatcherInput) MatcherResult`

## License

MIT
