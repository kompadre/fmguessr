# AGENTS.md

## Rules
- Read enough context before editing. Don't creep offsets line-by-line.
- If Edit fails twice, stop and explain the issue.
- Verify with `go build ./...`, `go vet ./...`, `go test` — not manual inspection.
- When the user says a bug is recurring or similar to a past fix, offer a regression test first. Tag with `//go:build regression`.
- Commit on task completion, once the user has verified the fix is correct.
- Never call `LoadCodebook()` or `LoadTimbreCodebook()` directly — use `loadCodebooksOnce()` in `matcher.go`.
- Wrap errors: `fmt.Errorf("context: %w", err)`.
- Match existing code style in each file. Don't introduce conventions that conflict with neighbors.
- Use `pianoDockWrapper` (module-scoped in app.js) to access `_voiceBank`, `_pianoroll`, `_showLog` — NOT `document.getElementById('piano-dock')` which returns the outer container, not the inner wrapper.
- VoiceBank assigns colors by slot index (PATCH_COLORS[8]). Pianoroll receives color via `setActivePatch(id, data, color, channel)` — always pass `entry.color` from voicebank.
- Verify JS changes with `node --check html/app.js`, `node --check html/components/*.js`, `node --check html/synth/*.js`.
- WASM build: `make -f Makefile.msfa` in `html/synth/wasm/` (uses `zig c++ -target wasm32-wasi`). Do NOT use emscripten.
- Never export `malloc`/`free` from WASM. Use static pre-allocated buffers (`get_patch_buffer()`, `load_voice_from_buffer()`).
- AudioManager is a singleton — `initAudio()` runs once. Never re-call `addModule()` after initial setup.
- Shutdown is disabled for testing — `shutdownAll()`, `setPianorollStop()`, `extendIdle()` are no-ops.

## Git & Version Control Rules

- **NEVER alter Git history.** Under no circumstances should you run `git rebase`, `git commit --amend`, `git reset`, `git cherry-pick`, or perform any commit squashing/history modifications.
- **Append-only commits:** All changes must strictly be committed as NEW, additive commits on top of the current branch.
- **No speculative or broken commits:** Only commit when the code is verified, working, and builds/runs without errors.
- **Clear commit messages:** Write short, descriptive commit messages summarizing the actual functional changes made.

## Commands

```bash
make                    # Build all binaries
make test               # Unit + E2E tests
make test-smoke         # Fast e2e smoke tests (no external fixtures)
make test-regression    # Regression tests (requires -tags regression)
go test -v ./pkg/match/...           # Unit tests only
go test -v -tags regression ./tests/...  # Regression tests only
go test -race ./pkg/match/...        # Race detector
```

## Architecture

`fmguessr` matches DX7 FM patches to target audio via genetic algorithms.

**Pipeline:** Codebook KNN → Formant candidates → NN prediction → Classifier → GA (32-island niche → global convergence → settle → hill-climbing → formant refinement)

**Key constraints:**
- DX7 patch: 128 bytes, Op6 first. `Encode()` overlays tuned params onto `RawBase`, preserving LFO/PitchEnv/Scaling.
- `RenderCandidate` creates a fresh MSFA synth per call. `evaluateBatchParallel` uses `GOMAXPROCS` workers.
- `Candidate.Fitness = math.MaxFloat64` means unevaluated (zero-value = false perfect match).
- `microMutate`/`envelopeMutate` must receive `rng` (constant seed kills diversity).
- `cachedTargetMel` is local to `Run()`, passed to `evaluateBatchParallel` — never global.
- Render duration capped at 3.0s. Target WAV truncated on load.
- REST API: single-worker queue, 5 jobs max, 800KB upload limit.

**Distance metrics:**
- Primary: `MelCosineDistanceCached` — cosine on 128×128 mel spectrogram (16384-dim).
- Formant: spectral envelope peak matching (greedy nearest-neighbor, freq+amp). Used for candidate generation and post-GA refinement.
- Trajectory: dormant — infrastructure exists but auto-disables (scale mismatch with mel).

**Formant synthesis:**
- `ExtractFormants`: cepstral envelope + peak-picking → F1/F2/F3/F4 in Hz.
- `PredictSpectralPeaks`: Bessel function FM sideband prediction from DX7 patch params.
- `GenerateFormantCandidates`: maps target formants to DX7 operator ratios.
- `FormantRefine`: constrained optimization — minimizes formant distance while keeping mel ≤ 110%.
- Probe endpoint returns `F1Hz`, `F2Hz`, `F3Hz`.

**GA convergence:**
- Absolute threshold (0.001), 5-gen patience, min 20 global gens.
- Settle phase: 5 quiet generations to confirm convergence (no injection).
- Line search: batched into single `evaluateBatchParallel` call.
- Top-32 export: diverse assembly from island bests + formant + NN + GA.

**Caching:**
- Mel filterbank: `getMelFilterbank()` via `sync.Once`.
- Hann windows: `getHannWindow()` via `sync.Map`.
- Elite signatures: `EliteCohort.TryInsert` caches `Encode()` results.
- Session files: `exports/sessions/<hash>.bin`, atomic write, pruned after 1h.

**Frontend (Vanilla JS, no framework):**
- `html/app.js`: main orchestrator (~600 lines). Wires file input, menu bar, canvas, toolbar. PlayNote/playPatch with multi-channel support.
- `html/components/canvas.js`: Canvas — patches + PatchContainers on night-sky. Drag-to-reposition, snap-into-container, resize containers. State derived from linked Job.
- `html/components/toolbar.js`: Bottom toolbar — Keyboard | Piano Roll | Edit | Log tabs. Replaces old piano-dock.js.
- `html/components/edit-panel.js`: Morph / Params / Random editing. Extracted from old app.js.
- `html/components/job.js`: Job class — steering object for WAV import lifecycle. Owns: probe → match → poll → complete/fail. Renders probe form inside its container.
- `html/components/patch-model.js`: Unified data model + stores (localStorage). `patchStore`, `canvasStore`, `jobStore`.
- `html/components/patch-row.js`: Unified patch row rendering with star drag handle, mini spectrogram, assign/edit/remove buttons.
- `html/components/patch.js`: Waveform helpers only — `decodeWavSamples`, `drawMiniWaveform`, `wavSampleCache`.
- `html/components/voicebank.js`: Voice palette (8 slots). Persists to localStorage. Each entry has static channel assignment (0-7).
- `html/components/pianoroll.js`: Sequencer. Notes store per-note `voiceData` and `channel`. Error counter stops after 3 failures.
- `html/synth/audio-manager.js`: Singleton AudioContext + AudioWorklet. `acquire()` creates once, `initAudio()` loads modules once. Multi-channel: `loadVoice()`, `playNoteOnChannel()`. No malloc/free — uses static WASM buffers.
- `html/synth/dx7-synth.js`: Batch waveform rendering only (main thread WASM). No realtime synth code.
- `html/synth/dx7-processor.js`: AudioWorkletProcessor wrapping MSFA DX7 engine. Handles `voice-channel`, `midi`, `patch` messages.
- WASM: `html/synth/wasm/msfa.wasm` (zig c++ build), `msfa.wasm.js` (encoded for AudioWorklet loading). Static buffers: `get_patch_buffer()`, `load_voice_from_buffer()`.

**Deploy:** `make deploy` stages to `deploy/app/`. Docker via `deploy/Dockerfile` + `deploy/docker-compose.yml`.
