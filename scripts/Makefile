# scripts/Makefile — Training pipelines for fmguessr ML models
#
# Usage:
#   make predictor       — Full MLP predictor pipeline (generate + train + export)
#   make encoder         — Full audio encoder pipeline (render + train + export)
#   make classifier      — Full patch classifier pipeline (render + train + export)
#   make all             — Train all models
#
#   make predictor-dataset — Generate predictor training dataset only
#   make predictor-train   — Train predictor only
#   make clean-all         — Remove all renders, datasets, and models

.PHONY: all clean-all \
	predictor predictor-dataset predictor-train predictor-export \
	encoder encoder-render encoder-train encoder-export \
	classifier classifier-render classifier-train

CODEBOOK ?= dx7_codebook.bin

# ──────────────────────────────────────────────
# MLP Predictor (timbre → DX7 parameters)
# ──────────────────────────────────────────────

predictor: predictor-dataset predictor-train predictor-export

predictor-dataset:
	@echo "=== Generating predictor dataset ==="
	python scripts/predictor/generate_dataset.py \
		--codebook $(CODEBOOK) \
		--output scripts/predictor/dataset.npz

predictor-train:
	@echo "=== Training MLP predictor ==="
	python scripts/predictor/train_model.py \
		--dataset scripts/predictor/dataset.npz \
		--output scripts/predictor/model_best.pth

predictor-export:
	@echo "=== Exporting predictor to ONNX ==="
	python scripts/predictor/reexport.py

# ──────────────────────────────────────────────
# Audio Encoder (1D ResNet, trajectory distance)
# ──────────────────────────────────────────────

encoder: encoder-render encoder-train encoder-export

encoder-render:
	@echo "=== Rendering encoder WAVs ==="
	python scripts/encoder/generate_encoder_dataset.py \
		--codebook $(CODEBOOK) \
		--outdir scripts/encoder/encoder_wavs

encoder-train:
	@echo "=== Training audio encoder ==="
	python scripts/encoder/train_audio_encoder.py \
		--datadir scripts/encoder/encoder_wavs \
		--output scripts/encoder/encoder.onnx \
		--epochs 100

encoder-export:
	@echo "(Encoder exports to ONNX during training)"

# ──────────────────────────────────────────────
# Patch Classifier (CNN, mel spectrogram)
# ──────────────────────────────────────────────

classifier: classifier-render classifier-train

classifier-render:
	@echo "=== Rendering classifier dataset ==="
	./classifier --codebook $(CODEBOOK) \
		--output scripts/classifier/dataset_cls.bin \
		--dur 3s

classifier-train:
	@echo "=== Training patch classifier ==="
	python scripts/classifier/train_classifier.py \
		--dataset scripts/classifier/dataset_cls.bin \
		--output scripts/classifier/patch_classifier.onnx

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

predictor-test:
	python scripts/predictor/g0.py --dataset scripts/predictor/dataset.npz

predictor-check:
	python scripts/predictor/quickcheck.py

all: predictor encoder classifier

clean-predictor:
	rm -rf scripts/predictor/renders scripts/predictor/dataset.npz scripts/predictor/model_best.pth scripts/predictor/model.onnx scripts/predictor/model_best.onnx

clean-encoder:
	rm -rf scripts/encoder/renders scripts/encoder/encoder_wavs scripts/encoder/encoder.onnx

clean-classifier:
	rm -rf scripts/classifier/dataset_cls.bin scripts/classifier/patch_classifier.onnx

clean-all: clean-predictor clean-encoder clean-classifier
