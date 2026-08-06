# AGENTS.md

## Rules
- Read enough context before editing. Don't creep offsets line-by-line.
- If Edit fails twice, stop and explain the issue.
- Verify with `go build ./...`, `go vet ./...`, `go test` — not manual inspection.
- When the user says a bug is recurring or similar to a past fix, offer a regression test first. Tag with `//go:build regression`.
- Do not commit unless the user explicitly asks. Batch changes until told to commit.
- Never call `LoadCodebook()` or `LoadTimbreCodebook()` directly — use `loadCodebooksOnce()` in `matcher.go`.
- Wrap errors: `fmt.Errorf("context: %w", err)`.
- Match existing code style in each file. Don't introduce conventions that conflict with neighbors.
- Never call `rebuildPianoDock()` — it was removed. Piano dock is static. Use `currentJob` mutable reference for playback context.
- Verify JS changes with `node --check html/app.js`.

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
- `html/app.js`: main orchestrator, tab management, polling, patch rows, batch rendering
- `html/components/piano-dock.js`: static dock — keyboard | pianoroll | voicebank | log tabs, created once at startup via `initPianoDock()`
- `html/components/pianoroll.js`: sequencer, `draw()` is idempotent, uses `requestAnimationFrame` loop, exports `_pianoroll` with `setActivePatch()`/`stopPlayback()`
- `html/components/voicebank.js`: voice palette, persists to localStorage (`fmguessr_voices`), calls `pianorollRef.setActivePatch()` on selection
- `html/components/patch.js`: patch list rows, `.matched` class for active patch highlight
- `html/components/night-sky.js`: floating patch display with galactic core spinner
- `html/synth/dx7-synth.js`: Web Audio API layer — `initDx7Synth()`, `playNoteRealtime()`, `renderOffline()`, batch waveforms via offline graph
- `html/synth/dx7-processor.js`: AudioWorkletProcessor wrapping MSFA DX7 engine
- Mutable globals: `activeSyxVoiceData`, `activePianorollPatchId`, `activeKeyboardNote`, `currentJob`
- Pianoroll notes store per-note `voiceData` (patch that was active when drawn)
- Night sky patches use `activeKeyboardNote` for playback; imported match results use `job.probeData.midi_note`

**Deploy:** `make deploy` stages to `deploy/app/`. Docker via `deploy/Dockerfile` + `deploy/docker-compose.yml`.
