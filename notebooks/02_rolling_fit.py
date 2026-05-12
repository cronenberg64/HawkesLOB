"""Rolling window Hawkes fit to capture time-varying excitation.

Fits the model on 30-minute sliding windows and saves the results
for animation.
"""

import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors

from src.data import load_lobster, EVENT_LABELS
from src.model import fit_hawkes
from src.viz import plot_excitation_heatmap, set_technical_style

def run_rolling_analysis(ts_dict: dict[str, np.ndarray], window_size: float = 1800.0, step_size: float = 600.0):
    """Run rolling fit and return list of alpha matrices."""
    t_max = max(v.max() for v in ts_dict.values())
    starts = np.arange(0, t_max - window_size, step_size)
    
    alphas = []
    times = []
    
    print(f"Running rolling fit over {len(starts)} windows...")
    for t_start in starts:
        t_end = t_start + window_size
        ts_slice = {}
        for label in EVENT_LABELS:
            ts = ts_dict[label]
            ts_slice[label] = ts[(ts >= t_start) & (ts <= t_end)] - t_start
        
        try:
            res = fit_hawkes(ts_slice, decay=1.0)
            alphas.append(res["alpha"])
            times.append(t_start)
            print(f"  Window {t_start/3600:.1f}h - {t_end/3600:.1f}h: ρ={res['spectral_radius']:.4f}")
        except Exception as e:
            print(f"  Failed at {t_start}: {e}")
            
    return np.array(times), np.array(alphas)


def animate_rolling_heatmap(times: np.ndarray, alphas: np.ndarray, labels: list[str], save_path: str):
    """Animate the excitation matrix over time with technical styling."""
    set_technical_style()
    fig, ax = plt.subplots(figsize=(9, 8))
    
    vmax = np.max(alphas) * 0.8 
    vmin = 1e-3
    
    im = ax.imshow(alphas[0], cmap="magma", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(im, ax=ax, label="Excitation weight α")
    cb.outline.set_visible(False)
    
    texts = []
    for i in range(len(labels)):
        row_texts = []
        for j in range(len(labels)):
            t = ax.text(j, i, "", ha="center", va="center", fontweight="bold")
            row_texts.append(t)
        texts.append(row_texts)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")
    
    title = ax.set_title("", loc='left', pad=10)

    def update(frame):
        alpha = alphas[frame]
        t = times[frame]
        im.set_data(alpha)
        title.set_text(f"Regime Shift: {t/3600:.2f}h since market open")
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = alpha[i, j]
                texts[i][j].set_text(f"{val:.2f}")
                texts[i][j].set_color("white" if val > vmax * 0.4 else "grey")
        
        return [im, title] + [item for sublist in texts for item in sublist]

    ani = FuncAnimation(fig, update, frames=len(times), blit=True)
    print(f"Saving technical rolling heatmap to {save_path}...")
    ani.save(save_path, writer=PillowWriter(fps=3))
    plt.close()

if __name__ == "__main__":
    ts = load_lobster('data/GOOG_2012-06-21_34200000_57600000_message_10.csv')
    
    # Load or run analysis
    try:
        data = np.load("outputs/rolling_results.npz")
        times, alphas = data["times"], data["alphas"]
        print("Loaded existing rolling results.")
    except:
        times, alphas = run_rolling_analysis(ts)
        np.savez("outputs/rolling_results.npz", times=times, alphas=alphas)
    
    animate_rolling_heatmap(times, alphas, EVENT_LABELS, "outputs/rolling_heatmap.gif")
