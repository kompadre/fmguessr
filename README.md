# fmguessr

6-operator FM synthesizer patch generator and matcher. Uses genetic algorithms, CNN mel-spectrogram classification, MLP prediction, and live hill-climbing to match target audio WAV files and generate DX7 `.syx` bank files.

## Features

- **CNN Patch Classifier:** 128×128 mel spectrogram → 4-layer CNN → identifies DX7 presets from codebook. Preserves LFO vibrato/tremolo patterns.
- **MLP Predictor (~2ms):** Predicts continuous DX7 parameters from 256-bin timbre spectrum.
- **Preset Codebook:** KNN lookup via timbre distance for instant GA seeding.
- **32-Island GA:** Niche evolution across algorithm topologies → global convergence.
- **LiveOptimizer:** Multi-threaded hill-climbing with acceleration trough cacher and topological jumps.
- **Template Preservation:** LFO, pitch envelopes, and scaling from matched presets preserved.

## Build

```bash
make          # Build all tools
make test     # Run tests
```

Requires Go 1.20+, C++ compiler (MSFA engine), Python 3.8+ with PyTorch (for training).

## Usage

### Match a Target

```bash
# Standard GA matching
./match -v target.wav -syx result.syx

# With NN predictor
./match -model model.onnx -v target.wav -syx result.syx

# Classifier-only (fast, no GA)
./match --classifier patch_classifier.onnx --classifier-only -v target.wav -syx result.syx
```

### Play/Export

```bash
./play result.syx 0 C3 -o match.wav
```

### Train the MLP Predictor

```bash
python scripts/generate_dataset.py --codebook dx7_codebook.bin
python scripts/train_model.py --dataset dataset.npz --output model.onnx
```

### Train the Patch Classifier

```bash
# Generate mel spectrogram dataset (renders each patch at 37 MIDI notes, 3s each)
./classifier --codebook dx7_codebook.bin --output dataset_cls.bin --dur 3s

# Train CNN classifier → ONNX
python scripts/train_classifier.py --dataset dataset_cls.bin --output patch_classifier.onnx
```

### Index a Codebook

```bash
./index Dexed_01.syx
```

### REST API

```bash
./server -port 8080 -model model.onnx -codebook dx7_codebook.bin
```

- `POST /api/probe` — Analyze WAV envelope/pitch instantly
- `POST /api/match` — Async matching (returns job ID, poll `GET /api/status/:id`)
- `GET /exports/*` — Static files with CORS

## Architecture

```
target.wav
  ├─→ CNN Classifier (mel spectrogram → patch prediction)
  ├─→ KNN Timbre Codebook Search
  ├─→ [Optional] MLP Predictor (~2ms)
  ├─→ LiveOptimizer (acceleration leaps + topological jumps)
  └─→ 32-Island Niche GA
        └─→ Early termination at distance < 10⁻⁶
```

## License

MIT
