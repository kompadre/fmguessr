# fmguessr

`fmguessr` is an FM synthesis patch generator and matcher. It uses a hybrid genetic algorithm and audio fingerprinting system to match target audio WAV files and generate DX7-compatible System Exclusive (`.syx`) bank files.

## Features

- **Audio Fingerprinting:** Compact 32-dimensional perceptual timbre vector using MFCCs and spectral features.
- **Pitch-Invariant Matching:** Normalizes spectral features relative to fundamental frequency, allowing matching across different octaves.
- **Preset Codebook:** Fast K-Nearest Neighbors (KNN) lookup to seed optimization with known DX7 presets, drastically accelerating convergence.
- **Multi-Core Optimization:** Thread-safe parallel rendering using the C++ MSFA (Music Synthesizer For Android) engine.
- **Template Preservation:** Maintains non-tuned DX7 parameters (LFO, Pitch Envelopes) from matched presets for high-fidelity reconstruction.

## Installation

### Prerequisites

- Go 1.20+
- C++ compiler (for MSFA engine)

### Build

```bash
# Build all tools using Makefile
make

# Or build manually
go build -o match ./cmd/match/main.go
go build -o index ./cmd/index/main.go
go build -o play ./cmd/play/main.go
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
```

### 2. Match a Target Audio File

Find the best DX7 parameters to replicate a WAV file:

```bash
./match -v target.wav -syx result.syx
```

### 3. Play or Export the Match

Render the generated patch to verify the sound:

```bash
./play result.syx 0 C3 -o match.wav
```

## License

MIT License. See `MIT LICENCE.txt` for details.
