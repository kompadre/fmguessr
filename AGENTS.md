# AGENTS.md

# OpenCode Execution & Go Refactoring Rules

## 1. Tool Call Budget & Loop Prevention
- **No Micro-Read Loops:** NEVER issue consecutive `Read` tool calls with small line limits (e.g., 10–30 lines) or creeping offsets to verify an edit. Read entire functions or logical blocks in a single pass.
- **Fail Fast:** If an `Edit` tool call fails twice consecutively on the same target block due to line offset or string matching errors, STOP immediately. Do NOT attempt to read the file in small chunks to re-align. Explain the issue to the user or ask for clarification.
- **Whole-Function Replacements:** Prefer replacing entire functions or distinct `type` / `struct` definitions over multi-line search-and-replace strings. This prevents fuzzy-matching failures and line drift.
- **Use Diagnostics over Manual Inspection:** Use `go vet`, `go test`, or `golangci-lint` via the terminal tool to verify file state rather than manually inspecting file lines with repeated `Read` calls.

## 2. Go Refactoring & Code Style
- **Package Aliases:** Standardize package aliases (e.g., `rl` for Raylib). Treat package identifiers as compile-time constructs—never check package aliases or imports against `nil`.
- **Zero-Allocation Hot Paths:** In performance-critical loops (DSP, audio buffers, frame optimizations), avoid unnecessary heap allocations. Reuse slices/buffers where possible.
- **Error Handling:** Always wrap returned errors with context (e.g., `fmt.Errorf("live optimizer update: %w", err)`). Do not discard errors with blank identifiers `_` unless explicitly requested.
- **Explicit Struct & Field Naming:** Adhere strictly to consistent naming (e.g., use `Waveform` consistently rather than mixing `WaveForm`). Match existing structural patterns in the file.
- **Reference Semantics:** Remember that passing elements by reference in slices or maps allows in-place mutation; ensure pointers are verified before dereferencing to prevent runtime panics.

## 3. Plan-to-Build Execution Workflow
- **Execute in Stages:** When implementing complex architectural changes (like optimizer loops or audio metrics), execute one discrete logical step at a time.
- **Verification Gate:** After executing a single step, run `go check` or `go test ./...` to verify syntax and compilation before proceeding to the next file edit.

## Repository Overview
`fmguessr` is a 6-operator FM synthesizer patch generator that uses genetic algorithms, continuous MLP neural network prediction, and live hill-climbing optimization to match target audio WAV files and generate DX7-compatible System Exclusive (`.syx`) bank files.

## Developer Commands & Verification
- **Build all tools:** `make`
- **Run functional tests:** `make test`
- **Generate preset index:** `./index Dexed_01.syx`
- **Run matching optimization:** `GOMAXPROCS=16 ./match -d 1s -generations 30 -population 128 -syx result.syx -v target.wav`
- **Run matching with NN predictor:** `./match -model model.onnx -v target.wav`
- **Play/Export generated `.syx` patch:** `./play result.syx 0 C3 -o test.wav`
- **Generate training dataset:** `python scripts/generate_dataset.py --codebook dx7_codebook.bin`
- **Train NN model:** `python scripts/train_model.py --dataset dataset.npz --output model.onnx`

## Crucial Architecture & Memory Layout Facts

### 1. DX7 Bulk SysEx Layout & Template Preservation
- In a 128-byte DX7 voice patch, **Operator 6 is stored first** (bytes `0..16`) and **Operator 1 is stored last** (bytes `85..101`).
- `Candidate.Encode()` packs 50 parameters ($1 \text{ Algo} + 48 \text{ Op params} + 1 \text{ Feedback}$) into the 128-byte layout.
- **Template Preservation:** `Candidate.RawBase` (128 bytes) stores the full original patch template. `Encode()` overlays tuned parameters onto this template, preserving LFO, Pitch Envelopes, and Keyboard Scaling bytes for perfect timbre reconstruction.

### 2. Carrier Identification & Algorithm Table
- `IsCarrier(algo, opIdx)` queries `algOps[32][6]` (defined in `pkg/match/patch.go`) using bitwise check `(algOps[algo][opIdx] & 3) == 0`.
- Do not hardcode operator indices as carriers; always use `IsCarrier(algo, opIdx)`.

### 3. C++ MSFA Engine Concurrency & Determinism
- `external/msfa/synth_unit.h` uses per-instance `channel_outputs_` buffers to support multi-threaded evaluation.
- `evaluateBatchParallel` in `pkg/match/matcher.go` spawns worker goroutines matching `runtime.GOMAXPROCS`.
- **Acoustic Consistency:** Candidates MUST be evaluated via `RenderCandidate` (calling C++ `GetSamples`), which applies the `ResoFilter` and clipping. Pre-filter evaluation (operator sum) leads to spectral distance mismatch.
- **Non-determinism fixes:** `Lfo::unit_`, `PitchEnv::unit_`, `Lfo::phase_`, `Lfo::delaystate_`, `Lfo::randstate_`, and `Dx7Note::fb_buf_[2]` must all be instance members and reset in `reset()`/`init()`.

### 4. Timbre Extraction & Frame-Weighted Spectrum Distance
- `pkg/match/timbre.go` extracts sustained & percussive timbre features:
    - Automatically detects active sound duration. For short percussive samples ($T_{active} < 350\text{ms}$, e.g. hi-hats), window extraction starts at the active transient ($0\text{ms}$ skip) over $T_{active}$.
    - For sustained sounds, skips initial 80ms attack transient and identifies a 300ms window with minimum spectral flux.
    - Applies temporal median STFT and normalizes frame energy into a 256-bin log-frequency STFT spectrum (0–8kHz).
- `SpectralDistance` (`pkg/match/metric.go`) weights STFT frames by target RMS energy ratio ($w_i = \max(0.1, \text{RMS}_i/\text{RMS}_{max})$), preventing trailing silence from swamping percussive sound matching or favoring silent candidates.
- `Codebook` (`pkg/match/codebook_timbre.go`) indexes presets via timbre spectrum distance with goroutine workers (`IndexSysexBankTimbre`), providing instant KNN seeding for GA islands.

### 5. Timbre-First Pipeline & MLP Predictor
- `pkg/match/predictor.go` loads an ONNX MLP model for predicting continuous DX7 parameters (fine frequency, output level, feedback) directly from the 256-bin timbre spectrum.
- Training pipeline: `scripts/generate_dataset.py` (renders clean sustain WAVs) → `scripts/train_model.py` (PyTorch MLP with MSE loss → ONNX export).
- In the pipeline, NN continuous predictions seed algorithm candidates across all 32 islands before GA optimization.

### 6. 32-Island Niche Model & Adaptive Mutation
- Initial population seeds 32 algorithm sub-populations (islands) for niche evolution ($0..20$ generations) to prevent early algorithm extinction.
- Phase 2 merges top island candidates into a global population pool for final convergence.
- Step magnitudes scale dynamically with distance (`bestDist > 1.0` triggers multi-step hyper-mutations).
- **Early termination:** Stops immediately when best distance drops below $10^{-6}$.
- **Stagnation tracking:** Detects plateaus (6 gen patience) and injects codebook mutants from `CodebookPool` with boosted mutation rate (0.7) for 2 generations.
- **RMS normalization:** Uses RMS-based normalization instead of peak to preserve relative dynamic range for loud samples.

### 7. LiveOptimizer Hill-Climbing
- `pkg/match/live_optimizer.go` performs post-GA and NN pre-optimization fine-tuning:
    - **Multi-Threaded Parallel Batch Sweeps:** Evaluates coarse (`step=4..5`) and fine (`step=1`) knob steps across all 19 knobs simultaneously using `evaluateBatchParallel` on all CPU cores (`~400 candidates` per pass, `~0.1–0.3s`).
    - **Acceleration Trough Cacher:** Calculates discrete 2nd-derivative acceleration scores ($d_{i-1} - 2 d_i + d_{i+1}$) on knob sweeps to store top 3 convex distance troughs per knob (`updateAccelerationCache`). Probe passes test cached troughs for multi-step leaps over parameter valleys.
    - **Per-Operator Local Isolation:** Caches sweet spots per knob individually (`Op1.Level`, `Op2.Fine`, etc.) and automatically resets the cache on algorithm topology changes.
    - **DX7 Piano 1 Base:** Replaces silent zero state with DX7 Piano 1 default base (`epiano` template bytes) as a neutral starting seed.
    - **Topological Algorithm Jumps:** Runs 32-algorithm parallel sweeps on entry and plateau to escape local minima and test alternative operator routings.
    - **Optimized Render Duration:** Uses 1.0s render duration during hill-climbing for 2x faster evaluation while maintaining 100% STFT spectral accuracy.

### 8. GA Correctness Invariants
- `Candidate.Fitness` must be `math.MaxFloat64` for unevaluated candidates (zero-value `0.0` falsely signals a perfect match).
- `microMutate` must receive `rng` as a parameter (re-seeding with constant `42` eliminates population diversity).
- `replaceWeakestSorted` must copy the fitnesses slice internally to avoid corrupting `globalGA.currentFit`.

### 9. Duration Capping & Target WAV Truncation
- Requested render duration (`-d` flag) is strictly capped at **3.0 seconds** max in `cmd/match/main.go`.
- Target WAV audio is truncated up front upon loading to match the requested render duration, eliminating hangs and STFT computation overhead on long audio files.

### 10. REST API Server (`cmd/server`) & Queue Architecture
- **Single-Worker Execution:** Processes matching jobs sequentially from a Go channel (`jobQueue`) to prevent CPU thrashing during multi-threaded C++ rendering.
- **Queue Bounding:** Bounded queue (default: 5 pending jobs). Rejects requests with HTTP 429 / 503 when queue is full.
- **Connection Sanitization:** Uploads strictly limited to **800 KB** max using `http.MaxBytesReader` and instant TCP connection hijacker (`conn.Close()`) to drop oversized payloads. Decoded PCM is truncated to **4.0s max**.
- **Instant Probe Endpoint:** `POST /api/probe` analyzes WAV envelope ($A, D, S, R$), gate duration ($T_{\text{gate}}$), note-on/note-off positions, pitch/spectral heuristics, and checks if the patch was seen before (`seen_before: true` for distance $\le 0.001$).
- **Feature Overrides:** `POST /api/match` and `POST /api/probe` support form parameter overrides (`note` / `root_note` e.g. `"C3"` / `60`, `attack_ms`, `decay_ms`, `sustain_level`, `release_ms`, `gate_ms`, `brightness`, `harmonicity`).
- **Asynchronous Polling API:** `POST /api/match` returns HTTP 202 Accepted with job ID. `GET /api/status/:id` returns real-time progress (0–100%) and completed results (distances, top 5 match WAV URLs, `.syx` bank URL).
- **CORS-Enabled Exporter:** `GET /exports/*` serves generated candidate WAVs and `.syx` bank dumps with `Access-Control-Allow-Origin: *` for seamless frontend integration.
