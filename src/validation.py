"""Goodness-of-fit validation for the fitted Hawkes process.

Implements the time-rescaling theorem test and supplementary diagnostics:

1. **Time rescaling**: Under the true intensity, the compensator transforms
   Λᵢ(tₙ) − Λᵢ(tₙ₋₁) should be i.i.d. Exp(1). We compute these for the
   fitted model and test against Exp(1) via KS test.

2. **Ljung-Box**: Tests independence of the transformed inter-event times.
   If they are truly i.i.d. Exp(1), there should be no serial correlation.

3. **Spectral radius**: Already computed at fit time, but reported here too.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from .data import EVENT_LABELS


def _compute_intensity(t: float, mu_i: float, alpha_row: np.ndarray,
                       decay: float, all_timestamps: list[np.ndarray]) -> float:
    """Compute λᵢ(t) for dimension i at time t.

    λᵢ(t) = μᵢ + Σⱼ αᵢⱼ Σ_{tⱼₖ < t} β · exp(−β(t − tⱼₖ))
    """
    lam = mu_i
    for j, ts_j in enumerate(all_timestamps):
        # Only events strictly before t
        past = ts_j[ts_j < t]
        if len(past) > 0:
            lam += alpha_row[j] * decay * np.sum(np.exp(-decay * (t - past)))
    return lam


def _compute_compensator_increments(dim_idx: int, mu: np.ndarray,
                                     alpha: np.ndarray, decay: float,
                                     all_timestamps: list[np.ndarray]) -> np.ndarray:
    """Compute compensator increments Λᵢ(tₙ) − Λᵢ(tₙ₋₁) for dimension i.

    For an exponential kernel Hawkes process, the compensator between
    consecutive events of dimension i at times t_{n-1} and t_n is:

        Λᵢ(tₙ) − Λᵢ(tₙ₋₁) = μᵢ(tₙ − tₙ₋₁)
            + Σⱼ αᵢⱼ Σ_{tⱼₖ < tₙ} [exp(−β max(0, tⱼₖ − tₙ₋₁))  ... ]

    We use the closed-form integral of the exponential kernel between
    the two event times rather than numerical quadrature.

    For each source dimension j, the contribution between [a, b] is:
        αᵢⱼ · Σ_{tⱼₖ} ∫_a^b β·exp(−β(s − tⱼₖ)) · 1[tⱼₖ < s] ds
      = αᵢⱼ · Σ_{tⱼₖ < b} [exp(−β·max(0, a − tⱼₖ)) − exp(−β(b − tⱼₖ))]
    """
    ts_i = all_timestamps[dim_idx]
    n_events = len(ts_i)
    if n_events < 2:
        return np.array([])

    increments = np.empty(n_events - 1)

    for n in range(1, n_events):
        a = ts_i[n - 1]  # previous event time of dim i
        b = ts_i[n]      # current event time of dim i

        # Baseline contribution
        inc = mu[dim_idx] * (b - a)

        # Excitation contributions from each source dimension
        for j, ts_j in enumerate(all_timestamps):
            if alpha[dim_idx, j] == 0:
                continue
            # Events of dim j before time b
            mask = ts_j < b
            past_j = ts_j[mask]
            if len(past_j) == 0:
                continue

            # ∫_a^b αᵢⱼ β exp(−β(s − tⱼₖ)) ds for each tⱼₖ < b
            # = αᵢⱼ [exp(−β·max(0, a − tⱼₖ)) − exp(−β(b − tⱼₖ))]
            term_a = np.exp(-decay * np.maximum(0.0, a - past_j))
            term_b = np.exp(-decay * (b - past_j))
            inc += alpha[dim_idx, j] * np.sum(term_a - term_b)

        increments[n - 1] = inc

    return increments


def time_rescaling(fit_result: dict, timestamps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute time-rescaled inter-event times for each dimension.

    Under the true model, these should be i.i.d. Exp(1).

    Parameters
    ----------
    fit_result : dict
        Output of ``fit_hawkes``.
    timestamps : dict[str, np.ndarray]
        The same timestamps used for fitting.

    Returns
    -------
    dict[str, np.ndarray]
        Per-dimension arrays of transformed inter-event times τ.
    """
    mu = fit_result["mu"]
    alpha = fit_result["alpha"]
    decay = fit_result["decay"]
    labels = fit_result["labels"]

    all_ts = [timestamps[label] for label in labels]

    result = {}
    for i, label in enumerate(labels):
        print(f"  Computing compensator for {label} ({len(all_ts[i]):,} events)...")
        increments = _compute_compensator_increments(i, mu, alpha, decay, all_ts)
        result[label] = increments

    return result


def gof_test(transformed_times: dict[str, np.ndarray]) -> dict:
    """Run KS and Ljung-Box tests on transformed inter-event times.

    Parameters
    ----------
    transformed_times : dict[str, np.ndarray]
        Output of ``time_rescaling``.

    Returns
    -------
    dict
        Keys are event labels, values are dicts with:
        ``ks_stat``, ``ks_p``, ``lb_stat``, ``lb_p``
    """
    results = {}
    for label in EVENT_LABELS:
        tau = transformed_times[label]
        if len(tau) == 0:
            results[label] = {"ks_stat": np.nan, "ks_p": np.nan,
                              "lb_stat": np.nan, "lb_p": np.nan}
            continue

        # KS test against Exp(1)
        ks_stat, ks_p = stats.kstest(tau, "expon", args=(0, 1))

        # Ljung-Box on the transformed times (test for serial correlation)
        n_lags = min(10, len(tau) // 5)
        if n_lags < 1:
            n_lags = 1
        lb = acorr_ljungbox(tau, lags=n_lags, return_df=True)
        # Report the last lag's statistic and p-value
        lb_stat = lb["lb_stat"].iloc[-1]
        lb_p = lb["lb_pvalue"].iloc[-1]

        results[label] = {
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "lb_stat": float(lb_stat),
            "lb_p": float(lb_p),
        }

        status_ks = "✓ PASS" if ks_p > 0.05 else "✗ FAIL"
        status_lb = "✓ PASS" if lb_p > 0.05 else "✗ FAIL"
        print(f"  {label:5s}  KS: D={ks_stat:.4f}, p={ks_p:.4f} [{status_ks}]"
              f"  |  LB: Q={lb_stat:.2f}, p={lb_p:.4f} [{status_lb}]")

    return results
