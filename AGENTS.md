# AGENTS.md

## Repository Overview
`fmguessr` is a 6-operator FM synthesizer patch generator that uses genetic algorithms to match target audio WAV files and generate DX7-compatible System Exclusive (`.syx`) bank files.

## Developer Commands & Verification
- **Build all tools:** `make`
- **Run functional tests:** `make test`
- **Generate preset index:** `./index Dexed_01.syx`
- **Run matching optimization:** `GOMAXPROCS=16 ./match -d 1s -generations 30 -population 128 -syx result.syx -v target.wav`
- **Play/Export generated `.syx` patch:** `./play result.syx 0 C3 -o test.wav`

## Crucial Architecture & Memory Layout Facts

### 1. DX7 Bulk SysEx Layout & Template Preservation
- In a 128-byte DX7 voice patch, **Operator 6 is stored first** (bytes `0..16`) and **Operator 1 is stored last** (bytes `85..101`).
- `Candidate.Encode()` packs 50 parameters ($1 \text{ Algo} + 48 \text{ Op params} + 1 \text{ Feedback}$) into the 128-byte layout.
- **Template Preservation:** `Candidate.RawBase` (128 bytes) stores the full original patch template. `Encode()` overlays tuned parameters onto this template, preserving LFO, Pitch Envelopes, and Keyboard Scaling bytes for perfect timbre reconstruction.

### 2. Carrier Identification & Algorithm Table
- `IsCarrier(algo, opIdx)` queries `algOps[32][6]` (defined in `pkg/match/patch.go`) using bitwise check `(algOps[algo][opIdx] & 3) == 0`.
- Do not hardcode operator indices as carriers; always use `IsCarrier(algo, opIdx)`.

### 3. C++ MSFA Engine Concurrency & Filtering
- `external/msfa/synth_unit.h` uses per-instance `channel_outputs_` buffers to support multi-threaded evaluation.
- `evaluateBatchParallel` in `pkg/match/matcher.go` spawns worker goroutines matching `runtime.GOMAXPROCS`.
- **Acoustic Consistency:** Candidates MUST be evaluated via `RenderCandidate` (calling C++ `GetSamples`), which applies the `ResoFilter` and clipping. Pre-filter evaluation (operator sum) leads to spectral distance mismatch.

### 4. Audio Fingerprinting & Codebook KNN
- `pkg/match/fingerprint.go` extracts a **32-dimensional pitch-invariant vector**:
    - 10 harmonic energy ratios ($1\times..10\times F_0$).
    - 13 pitch-scaled MFCCs and $F_0$-normalized centroids.
- `Codebook` (`pkg/match/codebook.go`) provides instant KNN lookup for seeding the GA with known DX7 presets, bypassing hours of blind parameter guessing.

### 5. 32-Island Niche Model & Adaptive Mutation
- Initial population seeds 32 algorithm sub-populations (islands) for niche evolution ($0..20$ generations) to prevent early algorithm extinction.
- Phase 2 merges top island candidates into a global population pool for final convergence.
- Step magnitudes scale dynamically with distance (`bestDist > 1.0` triggers multi-step hyper-mutations).

