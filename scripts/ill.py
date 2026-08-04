import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP SIMULATION ENVIRONMENT & TARGET SPECTRAL LOSS FUNCTION
# =====================================================================
GRID_SIZE = 16  # 16 states per parameter (0 to 15)
TARGET = np.array([12, 10])  # The "ideal" knob settings in parameter space

def spectral_loss(state):
    """
    Simulates spectral distance to target audio.
    Lower score = closer to target sound.
    """
    # Euclidean distance with a mild non-linear harmonic distortion field
    diff = np.array(state) - TARGET
    dist = np.sqrt(np.sum(diff**2))
    # Add a slight ripple to simulate FM sideband non-linearities
    ripple = 0.3 * np.sin(state[0]) * np.cos(state[1])
    return dist + ripple

# Pre-compute loss surface for background visualization
loss_surface = np.zeros((GRID_SIZE, GRID_SIZE))
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        loss_surface[y, x] = spectral_loss([x, y])

# =====================================================================
# 2. NAIVE ODOMETER SEARCH (Full Unrolled Grid)
# =====================================================================
naive_path = []
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        naive_path.append([x, y])
        # Stop early if we hit target exactly
        if x == TARGET[0] and y == TARGET[1]:
            break
    else:
        continue
    break

naive_path = np.array(naive_path)
naive_evals = len(naive_path)

# =====================================================================
# 3. YOUR INFORMED DIRECTIONAL TRAJECTORY (Gradient/Attractor Unrolling)
# =====================================================================
start_state = np.array([1, 1])
trajectory = [start_state.copy()]

current_state = start_state.copy()
max_steps = 100

for _ in range(max_steps):
    current_loss = spectral_loss(current_state)
    
    # 1. Probe local tangent moves (+1/-1 step on each operator dial)
    probes = [
        current_state + np.array([1, 0]),   # Op1 +1
        current_state + np.array([-1, 0]),  # Op1 -1
        current_state + np.array([0, 1]),   # Op2 +1
        current_state + np.array([0, -1])   # Op2 -1
    ]
    
    # 2. Score candidate turns
    best_move = None
    best_loss = current_loss
    
    for candidate in probes:
        # Keep inside parameters bounds
        if 0 <= candidate[0] < GRID_SIZE and 0 <= candidate[1] < GRID_SIZE:
            loss = spectral_loss(candidate)
            if loss < best_loss:
                best_loss = loss
                best_move = candidate
                
    # 3. If no local single-step move improves loss, test momentum jump (+2)
    if best_move is None:
        momentum_probes = [
            current_state + np.array([2, 0]),
            current_state + np.array([-2, 0]),
            current_state + np.array([0, 2]),
            current_state + np.array([0, -2])
        ]
        for candidate in momentum_probes:
            if 0 <= candidate[0] < GRID_SIZE and 0 <= candidate[1] < GRID_SIZE:
                loss = spectral_loss(candidate)
                if loss < best_loss:
                    best_loss = loss
                    best_move = candidate
    
    # Stop if fully converged
    if best_move is None or np.array_equal(current_state, TARGET):
        break
        
    current_state = best_move
    trajectory.append(current_state.copy())

trajectory = np.array(trajectory)
informed_evals = len(trajectory) * 4  # ~4 probes per step

# =====================================================================
# 4. VISUALIZATION
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Naive Search
ax1.imshow(loss_surface, origin='lower', cmap='viridis_r', alpha=0.6)
ax1.plot(naive_path[:, 0], naive_path[:, 1], color='red', linestyle='-', marker='o', markersize=3, label='Odometer Path')
ax1.scatter(TARGET[0], TARGET[1], color='cyan', s=150, zorder=5, marker='*', label='Target Sound')
ax1.set_title(f"Naive Odometer Search\nEvaluations required: {naive_evals}", fontsize=12, fontweight='bold')
ax1.set_xlabel("Op1 Parameter (0-15)")
ax1.set_ylabel("Op2 Parameter (0-15)")
ax1.legend()

# Plot Informed Directional Trajectory
ax2.imshow(loss_surface, origin='lower', cmap='viridis_r', alpha=0.6)
ax2.plot(trajectory[:, 0], trajectory[:, 1], color='magenta', linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Directional Trajectory')
ax2.scatter(start_state[0], start_state[1], color='white', s=100, zorder=5, label='Start State')
ax2.scatter(TARGET[0], TARGET[1], color='cyan', s=150, zorder=5, marker='*', label='Target Sound')
ax2.set_title(f"Target-Attracted Trajectory\nTotal Probes/Evaluations: ~{informed_evals}", fontsize=12, fontweight='bold')
ax2.set_xlabel("Op1 Parameter (0-15)")
ax2.set_ylabel("Op2 Parameter (0-15)")
ax2.legend()

plt.tight_layout()
plt.show()
