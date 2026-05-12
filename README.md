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
2.  **Liquidity Provision Feedback**: Market Buys trigger a massive surge in Limit Buy additions ($\alpha_{LA\_B, MB} > 1.0$), suggesting a "super-critical" local response where market makers aggressively restock the book after liquidity is consumed.
3.  **Regime Shifts**: Rolling 30-minute fits reveal that the **Spectral Radius** $\rho(\alpha)$ fluctuates significantly throughout the day, peaking during periods of high volatility where the system approaches a critical state.

![Excitation Matrix Heatmap](outputs/excitation_matrix.png)
*Full-day average excitation weights. Note the strong diagonal and the cross-excitation between Market Orders and Limit Additions.*

---

## Methodology

### 1. The Hawkes Model
We use a multivariate Hawkes process with exponential kernels to capture the conditional intensity:

$$\lambda_i(t) = \mu_i + \sum_{j} \alpha_{ij} \sum_{t_{jk} < t} \beta \exp \left[ -\beta (t - t_{jk}) \right]$$

*   **$\mu_i$**: Baseline intensity (exogenous arrivals).
*   **$\alpha_{ij}$**: Excitation weight (impact of event $j$ on type $i$).
*   **$\beta$**: Shared decay rate (inverse half-life of excitation memory).

### 2. Validation (Time-Rescaling Theorem)
To ensure the model is not just curve-fitting, we validate using the **Time-Rescaling Theorem**. Under the true model, the compensator-transformed inter-event times must be i.i.d. $Exp(1)$.
*   **KS Test**: We compare empirical quantiles against the theoretical Exponential distribution.
*   **Ljung-Box**: We verify serial independence in the residuals to ensure all self-excitation has been captured.

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
python scripts/patch_tick.py  # Critical: Fixes tick library for Python 3.12+
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