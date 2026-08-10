import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

# 1. Generate static curves centered precisely around zero (middle Y = 0.0)
def generate_zero_centered_curve(seed, num_points=1000):
    x = np.linspace(0, 9, num_points)
    rng = np.random.default_rng(seed)
    y = np.zeros_like(x)
    for _ in range(5):
        freq = rng.uniform(0.3, 2.0)
        amp = rng.uniform(5, 20)
        phase = rng.uniform(0, 2 * np.pi)
        y += amp * np.sin(2 * np.pi * freq * (x / 9) + phase)
    # Zero-center by subtracting the mean amplitude
    y = y - np.mean(y)
    return x, y

x_left, y_left = generate_zero_centered_curve(seed=101)
x_right, y_right = generate_zero_centered_curve(seed=202)

def get_y(x_arr, y_arr, pos):
    idx = int((pos / 9.0) * (len(x_arr) - 1))
    return y_arr[idx]

# New Objective Function: abs(Y1 + Y2)
def evaluate_total(k1, k2):
    y1 = get_y(x_left, y_left, k1)
    y2 = get_y(x_right, y_right, k2)
    return abs(y1 + y2)

# --- Layout Setup ---
fig = plt.figure(figsize=(12, 8), facecolor='#111111')
fig.canvas.manager.set_window_title('Abs(Y1 + Y2) Waveform Optimization')

ax_left = fig.add_axes([0.08, 0.45, 0.35, 0.48], facecolor='#181818')
ax_center = fig.add_axes([0.45, 0.45, 0.10, 0.48], facecolor='#111111')
ax_right = fig.add_axes([0.57, 0.45, 0.35, 0.48], facecolor='#181818')

ax_center.axis('off')

# Plot curves
ax_left.plot(x_left, y_left, color='#00ffcc', lw=1.5, alpha=0.7)
ax_right.plot(x_right, y_right, color='#ff0055', lw=1.5, alpha=0.7)

# Draw prominent Y=0.0 Centerlines
ax_left.axhline(0.0, color='#ffffff', linestyle=':', alpha=0.5, lw=1.2)
ax_right.axhline(0.0, color='#ffffff', linestyle=':', alpha=0.5, lw=1.2)

# Evaluated points history markers
scat_left = ax_left.scatter([], [], color='#00ffcc', s=40, alpha=0.4, label='Probed')
scat_right = ax_right.scatter([], [], color='#ff0055', s=40, alpha=0.4, label='Probed')

# Active Cursors
vl_left = ax_left.axvline(0, color='#ffffff', linestyle='--', alpha=0.8)
dot_left, = ax_left.plot(0, get_y(x_left, y_left, 0), 'o', color='#ffffff', ms=10, mec='#00ffcc', mew=2)

vl_right = ax_right.axvline(0, color='#ffffff', linestyle='--', alpha=0.8)
dot_right, = ax_right.plot(0, get_y(x_right, y_right, 0), 'o', color='#ffffff', ms=10, mec='#ff0055', mew=2)

for ax in [ax_left, ax_right]:
    ax.set_xlim(0, 9)
    ax.set_xticks(range(10))
    ax.tick_params(colors='white')
    ax.grid(True, color='#2a2a2a', linestyle=':')
    for spine in ax.spines.values():
        spine.set_color('#444444')

# Center Readout
number_text = ax_center.text(0.5, 0.60, "0.0", color='#ffffff', fontsize=32, fontweight='bold', ha='center', va='center')
formula_text = ax_center.text(0.5, 0.42, "|Y1 + Y2|", color='#888888', fontsize=11, ha='center', va='center')
status_text = ax_center.text(0.5, 0.28, "Step: 0", color='#00ffcc', fontsize=11, ha='center', va='center')

# --- Optimization Logic Generators ---
def grid_search_sequence():
    """Exhaustive Search: Evaluates all 100 combinations."""
    for k1 in range(10):
        for k2 in range(10):
            yield (k1, k2)

def coordinate_ascent_sequence():
    """Independent Coordinate Ascent: Sweep L, pick max, sweep R."""
    best_k1 = 0
    best_val = evaluate_total(best_k1, 0)
    for k1 in range(10):
        val = evaluate_total(k1, 0)
        yield (k1, 0)
        if val > best_val:
            best_val = val
            best_k1 = k1
            
    best_k2 = 0
    for k2 in range(10):
        val = evaluate_total(best_k1, k2)
        yield (best_k1, k2)
        if val > best_val:
            best_val = val
            best_k2 = k2

# State management
state = {
    'generator': coordinate_ascent_sequence(),
    'history_l': [],
    'history_r': [],
    'step_count': 0,
    'best_val': -1.0,
    'best_pos': (0, 0)
}

# --- UI Controls ---
ax_radio = fig.add_axes([0.08, 0.10, 0.28, 0.22], facecolor='#181818')
radio = RadioButtons(ax_radio, ('Coordinate Ascent (20 Steps)', 'Exhaustive Grid Search (100 Steps)'), active=0, activecolor='#00ffcc')
for text in radio.labels:
    text.set_color('white')

ax_btn_step = fig.add_axes([0.40, 0.18, 0.15, 0.08], facecolor='#222222')
ax_btn_reset = fig.add_axes([0.57, 0.18, 0.15, 0.08], facecolor='#222222')

btn_step = Button(ax_btn_step, 'STEP ▶', color='#00ffcc', hovercolor='#00cca3')
btn_reset = Button(ax_btn_reset, 'RESET ↺', color='#ff0055', hovercolor='#cc0044')

def set_position(k1, k2):
    y1 = get_y(x_left, y_left, k1)
    y2 = get_y(x_right, y_right, k2)
    
    vl_left.set_xdata([k1, k1])
    dot_left.set_data([k1], [y1])
    
    vl_right.set_xdata([k2, k2])
    dot_right.set_data([k2], [y2])
    
    # Calculate absolute sum result
    current_total = evaluate_total(k1, k2)
    number_text.set_text(f"{current_total:.1f}")
    
    if current_total > state['best_val']:
        state['best_val'] = current_total
        state['best_pos'] = (k1, k2)
        number_text.set_color('#00ffcc')  # Highlight new max
    else:
        number_text.set_color('#ffffff')
        
    state['history_l'].append((k1, y1))
    state['history_r'].append((k2, y2))
    
    scat_left.set_offsets(np.array(state['history_l']))
    scat_right.set_offsets(np.array(state['history_r']))
    
    status_text.set_text(f"Step: {state['step_count']}\nBest: {state['best_val']:.1f}")
    fig.canvas.draw_idle()

def on_step(event):
    try:
        k1, k2 = next(state['generator'])
        state['step_count'] += 1
        set_position(k1, k2)
    except StopIteration:
        status_text.set_text(f"FINISHED!\nMax: {state['best_val']:.1f} at ({state['best_pos'][0]},{state['best_pos'][1]})")

def on_reset(event=None):
    choice = radio.value_selected
    if 'Exhaustive' in choice:
        state['generator'] = grid_search_sequence()
    else:
        state['generator'] = coordinate_ascent_sequence()
        
    state['history_l'].clear()
    state['history_r'].clear()
    state['step_count'] = 0
    state['best_val'] = -1.0
    scat_left.set_offsets(np.empty((0, 2)))
    scat_right.set_offsets(np.empty((0, 2)))
    status_text.set_text("Step: 0")
    set_position(0, 0)

btn_step.on_clicked(on_step)
btn_reset.on_clicked(on_reset)
radio.on_clicked(on_reset)

on_reset()
plt.show()
