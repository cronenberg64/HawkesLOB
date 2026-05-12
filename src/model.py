"""Multivariate Hawkes process fitting with exponential kernels.

Uses tick.hawkes.HawkesExpKern to fit a 4-dimensional Hawkes process
to limit order book event streams. The model is:

    λᵢ(t) = μᵢ + Σⱼ αᵢⱼ Σ_{tⱼₖ<t} β · exp(−β(t − tⱼₖ))

where μᵢ ≥ 0 is the baseline intensity for dimension i, αᵢⱼ ≥ 0 is the
excitation weight from dimension j to dimension i, and β > 0 is the
shared exponential decay rate.

Stability requires the spectral radius ρ(α) < 1.
"""

import numpy as np
from tick.hawkes import HawkesExpKern

from .data import EVENT_LABELS


def fit_hawkes(timestamps: dict[str, np.ndarray], decay: float = 1.0) -> dict:
    """Fit a multivariate Hawkes process with shared exponential decay.

    Parameters
    ----------
    timestamps : dict[str, np.ndarray]
        Output of ``load_lobster``. Keys are event labels, values are
        sorted 1-D timestamp arrays (seconds relative to market open).
    decay : float
        Shared exponential decay parameter β (inverse seconds).
        Default is 1.0 — a reasonable starting point for LOB data at
        ~second resolution. Tune via cross-validation or BIC if needed.

    Returns
    -------
    dict
        ``"model"``  : the fitted ``HawkesExpKern`` object
        ``"mu"``     : np.ndarray of shape (4,), baseline intensities
        ``"alpha"``  : np.ndarray of shape (4, 4), excitation matrix
        ``"decay"``  : the decay parameter used
        ``"spectral_radius"`` : float, spectral radius of α
        ``"labels"`` : list of event labels in dimension order
    """
    dim = len(EVENT_LABELS)

    # tick expects a list of lists: outer = realizations, inner = dimensions
    # We have a single realization
    event_list = [timestamps[label] for label in EVENT_LABELS]

    # Decay matrix: shared β across all pairs
    decays = decay * np.ones((dim, dim))

    model = HawkesExpKern(decays, penalty="none", solver="agd",
                          max_iter=1000, tol=1e-8, verbose=False)
    model.fit([event_list])

    mu = model.baseline
    alpha = model.adjacency

    # Spectral radius: largest absolute eigenvalue of α
    eigvals = np.linalg.eigvals(alpha)
    spectral_radius = float(np.max(np.abs(eigvals)))

    if spectral_radius > 0.95:
        print(f"⚠️  WARNING: spectral radius = {spectral_radius:.4f} (>0.95). "
              f"Process is near-critical or unstable.")

    print(f"\nFitted Hawkes process (β = {decay})")
    print(f"  Spectral radius: {spectral_radius:.4f}")
    print(f"\n  Baseline intensities μ (events/sec):")
    for i, label in enumerate(EVENT_LABELS):
        print(f"    {label:5s}: {mu[i]:.4f}")

    print(f"\n  Excitation matrix α (row=target, col=source):")
    header = "       " + "  ".join(f"{l:>7s}" for l in EVENT_LABELS)
    print(header)
    for i, label in enumerate(EVENT_LABELS):
        row = "  ".join(f"{alpha[i, j]:7.4f}" for j in range(dim))
        print(f"    {label:5s} {row}")

    return {
        "model": model,
        "mu": np.array(mu),
        "alpha": np.array(alpha),
        "decay": decay,
        "spectral_radius": spectral_radius,
        "labels": list(EVENT_LABELS),
    }
