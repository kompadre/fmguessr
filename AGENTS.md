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
- VoiceBank channel = array index. Entries have no `channel` property. Slot IS the channel.
- Verify JS changes with `node --check html/app.js`, `node --check html/components/*.js`, `node --check html/synth/*.js`.
- WASM build: `make -f Makefile.msfa` in `html/synth/wasm/` (uses `zig c++ -target wasm32-wasi`). Do NOT use emscripten.
- Never export `malloc`/`free` from WASM. Use static pre-allocated buffers (`get_patch_buffer()`, `load_voice_from_buffer()`).
- AudioManager is a singleton — `initAudio()` runs once. Never re-call `addModule()` after initial setup.
- Shutdown is disabled for testing — `shutdownAll()`, `setPianorollStop()`, `extendIdle()` are no-ops.
- `evaluateBatchParallel` takes `ctx context.Context` as its first parameter. Always pass it through from the enclosing function.

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

### Matcher Pipeline

`Run()` in `matcher.go` executes a pipeline of `Matcher` steps. Each step receives a `MatcherInput` (target audio, features, config, cohort snapshot, candidate pool, shared state) and returns a `MatcherResult` (partial `ResultsCohort`, `EarlyReturn` flag, `Err`). After each step, `Reconcile()` merges results into the shared cohort. Steps can request early return (e.g., classifier high-confidence, exact codebook match).

```go
type Matcher interface {
    Name() string
    Run(ctx context.Context, in MatcherInput) MatcherResult
}
```

**Pipeline steps** (in order, each in its own `matcher_*.go` file):
1. `CodebookSearch` — dual codebook KNN (fingerprint + timbre), populates `State.KNNMap`
2. `FormantGenerate` — spectral envelope → DX7 candidates, populates `State.FormantPeaks`
3. `NNPredict` — MLP probe predictor, populates `State.NNCandidates`
4. `NNRefine` — LiveOptimizer + DiffOptimizer on best NN, populates `State.LiveOpt`
5. `BeamSearch` — constructive beam search, populates `State.BeamResult`
6. `CSPOAStep` — progressive operator attachment
7. `ClassifierStep` — ONNX CNN classifier, can early-return on high confidence
8. `GAStep` — 32-island niche GA → global convergence (uses Pool + KNNMap + NNCandidates + BeamResult + LiveOpt for seeding)
9. `HillClimbStep` — LiveOptimizer + DiffOptimizer on GA best
10. `FormantRefineStep` — formant-constrained refinement

**Key types** (`matcher_types.go`):
- `ResultsCohort` — Best + Timbral[5] + Full[5] + Formant[5] (fixed-size arrays, zero-allocation)
- `CandidatePool` — fixed 1024-slot pool of `Candidate` values
- `MatcherState` — shared mutable state between steps (NNCandidates, FormantPeaks, KNNMap, BeamResult, LiveOpt, etc.)
- `Reconcile(other ResultsCohort)` — sorted insert with dedup by `Encode()`

### Candidate Layout

- `RawBase [128]byte` — fixed-size DX7 patch template. `NoRawBase` (all zeros) = no template.
- `Scores [NumScoreTypes]float64` — indexed by `ScoreTimbre`, `ScoreTemporal`, `ScoreFormant`. `math.NaN()` = not scored.
- `EmptyCandidate` — zero-value sentinel for uninitialized pool/cohort slots.
- `nanScores()` — returns a Scores array initialized to NaN.
- `Fitness = math.MaxFloat64` means unevaluated.
- `Encode()` overlays tuned params onto `RawBase` (or `epiano` default), preserving LFO/PitchEnv/Scaling.

### Key constraints

- `RenderCandidate` creates a fresh MSFA synth per call. `evaluateBatchParallel` uses `GOMAXPROCS` workers and accepts `ctx context.Context` for mid-batch cancellation.
- `microMutate`/`envelopeMutate` must receive `rng` (constant seed kills diversity).
- `cachedTargetMel` is local to `Run()`, passed via `MatcherInput.TargetMel` — never global.
- Render duration capped at 3.0s. Target WAV truncated on load.
- REST API: single-worker queue, 5 jobs max, 800KB upload limit.

**MSFA DX7 synth architecture (multitimbral):**
- 16 MIDI channels, 32 patch slots, 16 polyphonic voices, 16 LFOs.
- Fixed channel→slot mapping: channel N always reads from slot N. Set at init via `ProgramChange(ch, ch)`.
- `SetVoice(slot, data)` → `memcpy` into `patch_data_[slot*128]` + `UnpackPatch` into `unpacked_patch_[slot]` + LFO reset. Channel uses new patch immediately on next note-on.
- `ProgramChange(p, ch)` → sets `current_patch_[ch] = p`, unpacks `patch_data_[p*128]`, resets LFO.
- Default patch: silent INIT (all output levels 0). All 32 slots initialized with it.
- Note-on reads `unpacked_patch_[cmd_ch]` where `cmd_ch = MIDI_status & 0x0f`.
- VoiceBank → synth via `sendVoiceSnapshot(entries)`: worklet receives `voice-snapshot` message, loops `loadPatch(data, slot)` per entry. No ProgramChange needed.
- Preview: offline render on main thread (`offlineWasm`) + sample injection into worklet (`play-samples` message). Uses slot 0 / channel 0.
- Pianoroll playback: `playNoteOnChannel(channel, midi)` via worklet `send_midi(0x90 | ch, ...)`. Channel = VoiceBank array index.

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
- Elite signatures: `ResultsCohort.Reconcile` dedup by `Encode()`.
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
- `html/synth/dx7-processor.js`: AudioWorkletProcessor wrapping MSFA DX7 engine. Handles `voice-snapshot`, `midi`, `patch`, `play-samples`, `channel-volume` messages.
- WASM: `html/synth/wasm/msfa.wasm` (zig c++ build). Binary fetched on main thread, passed to AudioWorklet via `processorOptions`. Static buffers: `get_patch_buffer()`, `load_voice_from_buffer()`. Voice stealing with 64-sample fade-out.

**Server security:**
- `safeSyxPath()` validates resolved paths stay within export directory (blocks path traversal).
- Job ID validated against `^job_\d{1,10}_\d{1,6}$` before filesystem access.
- CORS: localhost origins get `*`, non-localhost get reflected origin.
- Cleanup loop prunes `job_*`, `play_*`, `morph_*`, `export_*` directories after 1h.

**Deploy:** `make deploy` stages to `deploy/app/`. Docker via `deploy/Dockerfile` + `deploy/docker-compose.yml`.

**Training:** `scripts/Makefile` uses configurable `PYTHON ?= python3`. Override with `make PYTHON=/path/to/python3.11`.
