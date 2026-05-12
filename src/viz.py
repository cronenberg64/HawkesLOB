"""Visualization modules for Hawkes process dynamics.

Includes functions for:
1. Smooth intensity stream calculation.
2. Technical 2D Spectrograms (Intensity & LOB).
3. Excitation matrix analysis.
4. Impulse response (cascade) multiples.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd

# Technical Color Palette (Coherent & Legible)
EVENT_COLORS = {
    "MB": "#ff595e",   # Bright Red
    "MS": "#1982c4",   # Bright Blue
    "LA_B": "#8ac926", # Green
    "LA_S": "#ffca3a"  # Yellow
}
BG_COLOR = "#0d1117"   # GitHub Dark
TEXT_COLOR = "#c9d1d9"

def set_technical_style():
    """Apply high-end technical styling to Matplotlib."""
    plt.rcParams.update({
        "axes.facecolor": BG_COLOR,
        "figure.facecolor": BG_COLOR,
        "axes.edgecolor": "#30363d",
        "grid.color": "#30363d",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Source Sans Pro", "DejaVu Sans"],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "grid.alpha": 0.3
    })

def compute_intensity_grid(ts_dict: dict[str, np.ndarray],
                          fit_result: dict,
                          t_start: float,
                          t_end: float,
                          resolution: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Compute intensities λ_i(t) on a regular grid for visualization.
    Uses a recursive update for efficiency: O(N_events + N_grid).
    """
    mu = fit_result["mu"]
    alpha = fit_result["alpha"]
    decay = fit_result["decay"]
    labels = fit_result["labels"]
    dim = len(labels)

    t_grid = np.linspace(t_start, t_end, resolution)
    dt = t_grid[1] - t_grid[0]
    intensities = np.zeros((dim, resolution))

    R = np.zeros(dim)
    for j, label in enumerate(labels):
        ts = ts_dict[label]
        past = ts[ts < t_start]
        if len(past) > 0:
            R[j] = np.sum(np.exp(-decay * (t_start - past)))

    ptr = [np.searchsorted(ts_dict[l], t_start) for l in labels]

    for k, t in enumerate(t_grid):
        for i in range(dim):
            intensities[i, k] = mu[i] + np.sum(alpha[i] * decay * R)
        
        if k < resolution - 1:
            t_next = t_grid[k+1]
            R *= np.exp(-decay * dt)
            for j, label in enumerate(labels):
                ts = ts_dict[label]
                while ptr[j] < len(ts) and ts[ptr[j]] <= t_next:
                    R[j] += np.exp(-decay * (t_next - ts[ptr[j]]))
                    ptr[j] += 1

    return t_grid, intensities

def plot_excitation_heatmap(alpha: np.ndarray, labels: list[str], ax=None, save_path=None):
    """Plot a static heatmap of the excitation matrix alpha."""
    set_technical_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    vmin = max(alpha[alpha > 0].min() * 0.5, 1e-6) if (alpha > 0).any() else 1e-6
    vmax = alpha.max() * 1.2
    
    im = ax.imshow(alpha, cmap="YlOrRd", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = alpha[i, j]
            color = "white" if val > vmax * 0.3 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontweight="bold")
            
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")
    ax.set_title("Excitation Matrix α")
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    
    return im

def animate_intensity_stream(ts_dict: dict, fit_result: dict, t_start: float, duration: float, 
                             fps: int = 20, save_path: str = "outputs/intensity_stream.gif"):
    """Headline visualization: Stacked intensity stream with event ticks."""
    set_technical_style()
    t_end = t_start + duration
    t_grid, intensities = compute_intensity_grid(ts_dict, fit_result, t_start, t_end, resolution=1000)
    
    labels = fit_result["labels"]
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)

    lines = []
    for i, label in enumerate(labels):
        line, = ax_top.plot([], [], color=EVENT_COLORS[label], lw=2.0, label=label, alpha=0.9)
        lines.append(line)
    
    ax_top.legend(loc='upper right', frameon=False, ncol=4)
    ax_top.set_ylabel("Intensity λ(t)")
    ax_top.set_title("Inferred Hawkes Intensity Stream", loc='left', pad=15)
    
    for i, label in enumerate(labels):
        ts = ts_dict[label]
        ts_win = ts[(ts >= t_start) & (ts <= t_end)]
        ax_bot.vlines(ts_win, i - 0.3, i + 0.3, colors=EVENT_COLORS[label], lw=1.2, alpha=0.7)
    
    ax_bot.set_yticks(range(4))
    ax_bot.set_yticklabels(labels)
    ax_bot.set_xlabel("Seconds since market open")
    ax_bot.set_ylim(-0.5, 3.5)
    ax_bot.set_xlim(t_start, t_end)
    
    def update(frame):
        t_curr = t_start + (frame / (duration * fps)) * duration
        mask = t_grid <= t_curr
        for i in range(4):
            lines[i].set_data(t_grid[mask], intensities[i, mask])
        
        if np.any(mask):
            v_max = np.max(intensities[:, mask]) * 1.2
            ax_top.set_ylim(0, max(v_max, 0.2))
        return lines

    ani = FuncAnimation(fig, update, frames=int(duration * fps), blit=True)
    print(f"Saving intensity stream animation to {save_path}...")
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()

def plot_intensity_spectrogram(ts_dict: dict, fit_result: dict, t_start: float, duration: float, 
                              save_path: str = "outputs/intensity_spectrogram.png"):
    """Signal-processing aesthetic: Event types vs Time, intensity as color."""
    set_technical_style()
    t_end = t_start + duration
    t_grid, intensities = compute_intensity_grid(ts_dict, fit_result, t_start, t_end, resolution=800)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(np.log1p(intensities), aspect='auto', origin='lower',
                   extent=[t_start, t_end, -0.5, 3.5],
                   cmap='magma', interpolation='hanning')
    
    ax.set_yticks(range(4))
    ax.set_yticklabels(fit_result["labels"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Hawkes Intensity Spectrogram (Log-Scale)", loc='left')
    
    cb = plt.colorbar(im, ax=ax, label="log(1 + λ)")
    cb.outline.set_visible(False)
    
    print(f"Saving spectrogram to {save_path}...")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_bookmap(ob_file: str, msg_file: str, t_start: float, duration: float, 
                 save_path: str = "outputs/bookmap.png"):
    """Professional LOB visualization: Price vs Time Heatmap."""
    set_technical_style()
    t_end = t_start + duration
    
    ob = pd.read_csv(ob_file, header=None, nrows=150000)
    msg = pd.read_csv(msg_file, header=None, nrows=150000)
    msg.columns = ['time', 'type', 'id', 'size', 'price', 'dir']
    msg['time'] -= 34200.0
    
    mask = (msg['time'] >= t_start) & (msg['time'] <= t_end)
    idx = msg[mask].index
    ob_sub = ob.iloc[idx]
    times = msg.loc[idx, 'time'].values
    
    p_min = (ob_sub.iloc[:, 2].min() / 10000) - 0.05
    p_max = (ob_sub.iloc[:, 0].max() / 10000) + 0.05
    p_bins = np.linspace(p_min, p_max, 150)
    t_bins = np.linspace(t_start, t_end, 300)
    
    grid = np.zeros((len(p_bins), len(t_bins)))
    for i in range(len(times)):
        t_idx = np.searchsorted(t_bins, times[i]) - 1
        if t_idx < 0: continue
        for level in range(5):
            # Ask side
            p_a, v_a = ob_sub.iloc[i, level*4]/10000, ob_sub.iloc[i, level*4+1]
            p_idx = np.searchsorted(p_bins, p_a) - 1
            if 0 <= p_idx < len(p_bins): grid[p_idx, t_idx] += v_a
            # Bid side
            p_b, v_b = ob_sub.iloc[i, level*4+2]/10000, ob_sub.iloc[i, level*4+3]
            p_idx = np.searchsorted(p_bins, p_b) - 1
            if 0 <= p_idx < len(p_bins): grid[p_idx, t_idx] += v_b

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.log1p(grid), aspect='auto', origin='lower',
                   extent=[t_start, t_end, p_min, p_max],
                   cmap='turbo', interpolation='nearest')
    
    trades = msg[mask & (msg['type'] == 4)]
    ax.scatter(trades['time'], trades['price']/10000, color='white', s=5, alpha=0.5, label='Trades')
    
    ax.set_ylabel("Price ($)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Order Book Heatmap (Bookmap-style)", loc='left')
    ax.legend(frameon=False)
    
    cb = plt.colorbar(im, ax=ax, label="log(1 + Liquidity Volume)")
    cb.outline.set_visible(False)
    
    print(f"Saving bookmap to {save_path}...")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_cascade_multiples(fit_result: dict, duration: float = 3.0, 
                           save_path: str = "outputs/cascade_multiples.png"):
    """Mathematical transparency: 4x4 grid of impulse response curves."""
    set_technical_style()
    mu, alpha, decay, labels = fit_result["mu"], fit_result["alpha"], fit_result["decay"], fit_result["labels"]
    
    t = np.linspace(0, duration, 100)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), sharex=True, sharey='row')
    
    for i in range(4): # Target
        for j in range(4): # Source
            ax = axes[i, j]
            y = alpha[i, j] * decay * np.exp(-decay * t)
            ax.plot(t, y, color=EVENT_COLORS[labels[j]], lw=2)
            ax.fill_between(t, 0, y, color=EVENT_COLORS[labels[j]], alpha=0.2)
            
            if i == 0: ax.set_title(f"Source: {labels[j]}", fontsize=10)
            if j == 0: ax.set_ylabel(f"Target: {labels[i]}", fontsize=10)
            ax.grid(True, ls='--', alpha=0.3)

    plt.suptitle("Impulse Response Matrix (Kernel Decay)", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    print(f"Saving cascade multiples to {save_path}...")
    plt.savefig(save_path, dpi=200)
    plt.close()
