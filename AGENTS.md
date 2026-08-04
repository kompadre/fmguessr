# AGENTS.md

## Execution Rules
- **No Micro-Read Loops:** Read entire functions in one pass. Never creep offsets.
- **Fail Fast:** If Edit fails twice consecutively, STOP and explain the issue.
- **Whole-Function Replacements:** Prefer replacing entire functions over search-and-replace.
- **Verify with Diagnostics:** Use `go vet`, `go test`, not manual line inspection.

## Code Style
- Zero-allocation hot paths (DSP, audio buffers). Reuse slices.
- Wrap errors with context: `fmt.Errorf("context: %w", err)`.
- Consistent naming — match existing patterns in the file.

## Repository Overview
`fmguessr` is a 6-operator FM synthesizer patch generator using genetic algorithms, NN prediction, and live hill-climbing to match target audio WAV files and generate DX7 `.syx` bank files.

## Developer Commands
- **Build:** `make` | **Test:** `make test`
- **Index:** `./index Dexed_01.syx`
- **Match:** `GOMAXPROCS=16 ./match -d 1s -generations 30 -population 128 -syx result.syx -v target.wav`
- **Match with NN:** `./match -model model.onnx -v target.wav`
- **Classifier-only:** `./match --classifier patch_classifier.onnx --classifier-only -v target.wav -syx result.syx`
- **Play:** `./play result.syx 0 C3 -o test.wav`
- **Train probe predictor:** `python scripts/generate_dataset.py --codebook dx7_codebook.bin && python scripts/train_model.py --dataset dataset.npz --output model.onnx`
- **Train patch classifier:** `./classifier --codebook dx7_codebook.bin --output dataset_cls.bin --dur 3s && python scripts/train_classifier.py --dataset dataset_cls.bin --output patch_classifier.onnx`
- **Debug classifier:** `./classifier --codebook dx7_codebook.bin --dump-wav test.wav --dump-patch 0 --dur 3s`

## Architecture Facts

### 1. DX7 SysEx & Template Preservation
- 128-byte voice patch: Op6 first (bytes 0–16), Op1 last (bytes 85–101).
- `Candidate.RawBase` stores full original patch. `Encode()` overlays tuned params, preserving LFO/PitchEnv/Scaling.

### 2. MSFA Engine Concurrency
- Per-instance `channel_outputs_` buffers. `evaluateBatchParallel` spawns `GOMAXPROCS` workers.
- Must evaluate via `RenderCandidate` (applies ResoFilter + clipping).
- LFO/PitchEnv state must be instance members, reset in `reset()`/`init()`.

### 3. Timbre Extraction
- `timbre.go`: skips 80ms attack, finds 300ms min-flux window, temporal median STFT → 256-bin spectrum.
- `SpectralDistance`: weights frames by target RMS ratio to prevent silence swamping.

### 4. 32-Island GA & LiveOptimizer
- 32 algorithm islands for niche evolution → global convergence → parallel LiveOptimizer hill-climbing.
- LiveOptimizer: acceleration trough cacher, per-operator isolation, topological algorithm jumps.
- Early termination at distance < 10⁻⁶. Stagnation recovery via codebook mutants.

### 5. CNN Patch Classifier (Mel Spectrogram)
- **Architecture:** Conv2d 1→32→64→128→256 (BN+pool) → AdaptiveAvgPool(4×4) → FC 4096→512→128 → dual heads (MIDI 37 + patch N).
- **Input:** 128×128 mel spectrogram (`ComputeMelSpectrogram` in `spectrogram.go`). n_fft=2048, hop=512, f_min=20Hz, f_max=8kHz.
- **Training:** `./classifier --dur 3s` → `dataset_cls.bin` → `train_classifier.py` → ONNX with embedded patch bytes.
- **Inference:** `PatchClassifier` in `patch_classifier.go` loads ONNX, takes 16384-dim mel input, returns top-K patches.
- **Why mel over 784-dim:** Old features collapsed temporal modulation via median STFT. CNN on mel preserves LFO vibrato/tremolo patterns.
- **Render duration:** Must use `--dur 3s` to capture full LFO cycles.

### 6. Duration Capping & Server
- Render duration capped at 3.0s. Target WAV truncated on load.
- REST API: single-worker queue, bounded (5 jobs), 800KB upload limit, CORS-enabled exports.

## GA Invariants
- `Candidate.Fitness` = `math.MaxFloat64` for unevaluated (zero-value = false perfect match).
- `microMutate`/`envelopeMutate` must receive `rng` parameter (constant seed kills diversity).
- `replaceWeakestSorted` must copy fitnesses slice internally.
