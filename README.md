# HawkesLOB: Microstructure Self-Excitation Study

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A rigorous fit of a multivariate Hawkes process to limit order book (LOB) event data, featuring residual analysis via the time-rescaling theorem and high-density technical visualizations.

### Dynamic Intensity Stream
![Hawkes Intensity Stream](outputs/intensity_stream.gif)
*60-second snapshot of LOB events. Top: Stacked conditional intensities λᵢ(t). Bottom: Actual event tick marks (Market Buys, Market Sells, Limit Additions).*

---

## Summary & Key Findings

This project implements a 4-dimensional Hawkes process to quantify the self-exciting dynamics of high-frequency trading. By analyzing LOBSTER event logs for `GOOG`, we model how aggressive trades trigger immediate liquidity replenishment and clustering.

### Key Observations:
1.  **Strong Self-Excitation**: Market orders exhibit extreme clustering ($\alpha_{ii} \approx 0.85$), confirming that "trades beget trades" in high-frequency regimes.
2.  **Near-Critical Excitation**: Aggressive buys (MB) trigger strong excitation in Limit Buy additions ($\alpha_{LA\_B, MB} > 1.0$). While individual weights exceed 1.0, the system remains stationary with a full-matrix **Spectral Radius $\rho(\alpha) \approx 0.86$**.
3.  **Regime Shifts**: Rolling 30-minute fits reveal that $\rho(\alpha)$ fluctuates significantly throughout the day, peaking during periods of high volatility.

![Excitation Matrix Heatmap](outputs/excitation_matrix.png)
*Full-day average excitation weights. Note the strong diagonal and the cross-excitation between Market Orders and Limit Additions.*

---

## Methodology

### 1. The Hawkes Model
We use a multivariate Hawkes process with exponential kernels to capture the conditional intensity:

$$\lambda_i(t) = \mu_i + \sum_{j} \alpha_{ij} \sum_{t_{jk} < t} \beta \exp \left[ -\beta (t - t_{jk}) \right]$$

*   **$\mu_i$**: Baseline intensity (exogenous arrivals).
*   **$\alpha_{ij}$**: Excitation weight (expected number of secondary events).
*   **$\beta$**: Shared decay rate (inverse half-life of memory).

### 2. Validation & Results
Under the true model, the compensator-transformed inter-event times must be i.i.d. $Exp(1)$. We report the following goodness-of-fit metrics for the first hour of trading:

| Dimension | KS Stat | KS p-val | LB Stat | LB p-val | Verdict |
|:--- |:--- |:--- |:--- |:--- |:--- |
| **MB** | 0.4705 | < 0.001 | 23.61 | 0.0087 | Fail (D) |
| **MS** | 0.5226 | < 0.001 | 10.36 | 0.4096 | Pass (LB) |
| **LA_B** | 0.3421 | < 0.001 | 36.77 | < 0.001 | Fail (D, LB) |
| **LA_S** | 0.3689 | < 0.001 | 41.33 | < 0.001 | Fail (D, LB) |

### 3. Statistical Honesty & Limitations
The systematic failure of the Kolmogorov-Smirnov (KS) tests highlights the limitations of first-order exponential Hawkes models for LOB data:
*   **Kernel Misspecification**: Real LOB memory often follows a **Power Law** rather than a simple exponential decay.
*   **Missing Covariates**: The 4-dimensional model ignores **cancellations** and **mid-price returns**, which are critical drivers of intensity in real-world microstructure.
*   **Stationarity**: While the spectral radius $\rho < 1$ guarantees global stationarity, local "bursts" often exhibit temporary near-critical behavior that standard MLE struggles to capture.

---

## Technical Visualization Gallery

The repository prioritizes **legible, dense, and technical** 2D signals that provide actionable insights into market microstructure.

### 1. Intensity Spectrogram
Visualizes Hawkes conditional intensity as a time-frequency spectrogram, identifying bursts of cross-excitation activity across all 4 event types.
![Intensity Spectrogram](outputs/intensity_spectrogram.png)

### 2. Bookmap-style LOB Heatmap
A professional-grade liquidity landscape. Volume is log-scaled to reveal hidden depth, with actual trade events (dots) overlaid on the price-time grid.
![Bookmap Heatmap](outputs/bookmap.png)

### 3. Impulse Response Matrix
A 4×4 grid of small multiples surfacing the exact trigger-response dynamics ($G_{ij}(t)$) for every possible pair in the system.
![Cascade Matrix](outputs/cascade_multiples.png)

### 4. Animated Regime Shifts
A rolling window animation showing the evolution of the excitation matrix $\alpha$ over the trading day.
![Rolling Alpha](outputs/rolling_heatmap.gif)

---

## Setup & Reproducibility

### 1. Environment & Patching
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Critical: Fixes tick library for Python 3.12+ ABI compatibility. 
# This only modifies metaclass attribute management and does not affect MLE numerics.
python scripts/patch_tick.py
```

### 2. Data Acquisition
1. Download the `GOOG_2012-06-21` (10 levels) sample from [LOBSTER](https://lobsterdata.com/info/DataSamples.php).
2. Extract the `.csv` files into the `data/` directory.

### 3. Command Executions
```bash
# A. Initialize the analysis notebook
python notebooks/create_notebook.py

# B. Run global fit & generate primary visuals
python src/model.py

# C. Run rolling window analysis
python notebooks/02_rolling_fit.py
```

---

## Limitations & Future Work
1.  **Kernel Shape**: Replacing exponential with **Power Law (Pareto)** kernels for heavy-tailed memory.
2.  **Asymmetric Decay**: Modeling unique decay rates $\beta_i$ for each event type.
3.  **Price Impact**: Integrating price changes as a covariate to model the volatility-excitation feedback loop.

---

## License
Licensed under the [Apache License, Version 2.0](LICENSE).