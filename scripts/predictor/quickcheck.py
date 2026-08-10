import torch
import numpy as np

# 1. Import your model class from your training script
from train_model import DX7ParameterEstimator  # Replace with your actual model class name

# 2. Instantiate model & load weights
model = DX7ParameterEstimator()
state_dict = torch.load("model_best.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Load sample 0 from dataset.npz
data = np.load("dataset.npz")
sample_spec = data['spectrograms'][0]  # Shape is likely (N_MELS, N_FRAMES) or (1, N_MELS, N_FRAMES)

# Convert to float tensor and add batch & channel dims if needed (1, 1, N_MELS, N_FRAMES)
test_tensor = torch.from_numpy(sample_spec).float()
if test_tensor.ndim == 2:
    test_tensor = test_tensor.unsqueeze(0).unsqueeze(0)
elif test_tensor.ndim == 3:
    test_tensor = test_tensor.unsqueeze(0)

print(f"Input tensor shape: {test_tensor.shape}")

# 4. Run inference
with torch.no_grad():
    output = model(test_tensor)
    print("\n--- Model Output ---")
    print(output)
