# HawkesLOB: Microstructure Self-Excitation Study

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A rigorous fit of a multivariate Hawkes process to limit order book (LOB) event data, featuring residual analysis via the time-rescaling theorem and high-fidelity microstructure visualizations.

---

## Overview

This project implements a 4-dimensional Hawkes process to model the self-exciting dynamics of high-frequency trading events. By analyzing LOBSTER event logs, we quantify how aggressive trades trigger limit order replenishment and further clustering of market activity.

### Key Features
- **Multivariate Modeling**: Fits baseline intensities ($\mu$) and an excitation matrix ($\alpha$) for Market Buys, Market Sells, and Limit Additions.
- **Statistical Rigor**: Validates model fit using the **Time-Rescaling Theorem** with Kolmogorov-Smirnov (KS) tests and Ljung-Box residual checks.
- **Microstructure Insights**: Reveals the "refill" mechanism where liquidity consumption triggers immediate quote replenishment by market makers.

---

## Methodology

### The Model
We use a multivariate Hawkes process with exponential kernels:

$$\lambda_i(t) = \mu_i + \sum_j \alpha_{ij} \sum_{t_{jk}<t} \beta \cdot e^{-\beta(t-t_{jk})}$$

Where:
- $\mu_i$: Baseline intensity of event type $i$.
- $\alpha_{ij}$: Excitation weight (impact of event $j$ on type $i$).
- $\beta$: Shared exponential decay rate (inverse half-life of excitation).

### Validation (The Time-Rescaling Theorem)
To ensure the model isn't just "curve-fitting," we transform the inter-event times using the fitted intensity function. Under the true model, the transformed times $\tau$ should be i.i.d. $Exp(1)$. We test this using:
1. **KS Test**: Comparing empirical quantiles against the theoretical Exponential distribution.
2. **Ljung-Box**: Checking for serial correlation in the residuals to verify the process is truly Poisson.

---

## Setup & Reproducibility

### 1. Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Patching `tick`
The `tick` library requires a small patch for Python 3.12+ compatibility (included in this repo):
```bash
python scripts/patch_tick.py
```

### 3. Data Acquisition
This project uses **LOBSTER** sample data. For reproducibility, download the **GOOG** 10-level sample:
1. Go to [LOBSTER Data Samples](https://lobsterdata.com/info/DataSamples.php).
2. Download the `GOOG_2012-06-21` (10 levels) zip file.
3. Extract into the `data/` directory.

Expected files:
- `data/GOOG_2012-06-21_..._message_10.csv`
- `data/GOOG_2012-06-21_..._orderbook_10.csv`

---

## Analysis Walkthrough

The core analysis is contained in the Jupyter notebook:
- [`notebooks/01_fit.ipynb`](notebooks/01_fit.ipynb): Data loading, MLE fitting, and goodness-of-fit validation.

---

## License
Licensed under the [Apache License, Version 2.0](LICENSE).