import torch
from train_model import DX7ParameterEstimator  # Adjust import to match your class name

# 1. Load trained architecture
device = torch.device("cpu")
model = DX7ParameterEstimator()

# Load state dict (handles both raw state_dict and checkpoint dicts)
checkpoint = torch.load("model_best.pth", map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# 2. Dummy input matching 1D timbre spectrum dimensions (Batch=1, Ch=1, TimbreBins=256)
dummy_input = torch.randn(1, 1, 256, device=device)

# 3. Export to ONNX cleanly
output_onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_onnx_path,
    export_params=True,        # Store all weights directly inside the ONNX file
    opset_version=18,          # Match current ONNX spec
    do_constant_folding=True,  # Fold constant nodes for faster inference
    input_names=["spectrogram"],
    output_names=["continuous"],
    dynamo=False,              # Disable Dynamo to force clean legacy export
    dynamic_axes={
        "spectrogram": {0: "batch_size"},
        "continuous": {0: "batch_size"}
    }
)

print(f"Export successful!")
