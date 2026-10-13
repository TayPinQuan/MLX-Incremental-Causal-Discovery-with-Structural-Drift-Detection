# MLX Incremental Causal Discovery with Structural Drift Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, MLX-native framework for **incremental causal discovery** and
**structural drift detection** in time-series data. Designed for edge
deployment on Apple Silicon, this project implements an incremental version
of DirectLiNGAM that detects changes in causal structure and causal
coefficients, and efficiently updates the causal graph without full
retraining.

This is the first of two projects. The second project will extend the same
interfaces to **neural causal discovery**.

## Motivation

Industrial processes are non-stationary: equipment aging, raw material
changes, and recipe adjustments cause the underlying causal mechanisms to
drift over time. Traditional causal discovery assumes a static
data-generating process, and retraining from scratch on every window is
computationally prohibitive on edge devices. This project bridges that gap
by combining:

- **DirectLiNGAM** for identifiable causal structure learning (exploiting
  non-Gaussianity)
- **Sliding-window analysis** with delay embedding for lagged and
  instantaneous causality
- **Multi-signal structural drift detection** on residuals, coefficients,
  and graph structure
- **Incremental update strategies** (Markov-blanket local re-learning +
  warm start + consistency-checked fallback) to avoid full re-training
- **MLX** for unified memory and lazy computation on Apple Silicon

## Scope of the First Version

The first version focuses on depth rather than breadth:

- Observational time series only
- Lagged and instantaneous causality
- Approximately stationary within a window, drift allowed between windows
- Drift types: **structural drift** and **coefficient drift**
- Output: approximate graph with full-retrain fallback
- Deployment: single-machine on Apple Silicon

Out of scope for v1 (planned for v2 or the neural project):

- Noise drift, lag drift, periodic drift
- Latent confounders, non-linearity, feedback loops
- True sample-by-sample online updates
- Full-stack MLX implementation
- Causal ground-truth validation on real datasets

## Features

- MLX-native implementation of DirectLiNGAM core routines
  (least-squares regression, third-order cross-cumulants)
- Sliding window with configurable delay embedding to capture lagged causal
  effects
- Multi-signal drift score: residual energy distance + coefficient change
  + structural Hamming distance
- Double-sided CUSUM on the fused drift score, with adaptive threshold and
  permutation-test confirmation
- Markov-blanket local re-learning with warm start and consistency-checked
  fallback
- Visualization of causal graph evolution as an animated GIF
- Benchmarks against NumPy and PyTorch baselines for speed and memory

## Installation

Requires macOS on Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

```bash
conda create -n mlx-causal python=3.10
conda activate mlx-causal

pip install mlx
pip install numpy scipy networkx matplotlib pandas scikit-learn
pip install lingam  # reference implementation for validation
```

## Quick Start

### 1. Generate synthetic drifting data

```python
from data.synthetic import generate_drifting_data

data, meta = generate_drifting_data(
    n_samples=5000,
    n_vars=8,
    max_lag=2,
    drift_at=2500,
    drift_type='both',      # 'structure' | 'coefficients' | 'both'
    drift_mode='abrupt',    # 'abrupt' | 'gradual'
    random_state=42,
)
```

### 2. Run online causal discovery with drift detection

```python
from online_causal import OnlineCausalDiscovery

ocd = OnlineCausalDiscovery(
    window_size=500,
    max_lag=2,
    warmup=3,
    kappa=5.0,
    n_perm=200,
)

results = ocd.run(data)
```

### 3. Visualize causal evolution

```python
from visualize import animate_causal_evolution

animate_causal_evolution(
    results['adj_matrices'],
    drift_points=results['drift_points'],
    save_path='causal_evolution.gif',
)
```

## Method Overview

### Causal Discovery Engine

We implement a simplified DirectLiNGAM in MLX. The core independence
measure between residual and predictor uses third-order cross-cumulants:

```python
def lingam_residual_independence(X, i, j):
    xi, xj = X[:, i], X[:, j]
    beta = mx.linalg.lstsq(xj[:, None], xi[:, None])
    residual = xi - (xj * beta[0]).squeeze()
    residual = (residual - mx.mean(residual)) / (mx.std(residual) + 1e-8)
    xj_std = (xj - mx.mean(xj)) / (mx.std(xj) + 1e-8)
    return mx.abs(mx.mean(residual * xj_std * xj_std))
```

Delay embedding expands each variable into `max_lag + 1` copies. Lagged
edges are direction-fixed; instantaneous edges are learned by LiNGAM.
Coefficient estimation uses **target-prior Ridge** with the previous
window's coefficients as prior; edge selection uses backward elimination
with BIC.

### Drift Score

We fuse three continuous signals into one drift score:

1. **Residual energy distance** between consecutive windows, averaged over
   variables
2. **Relative coefficient change**
   `||B_t - B_{t-1}||_F / (||B_{t-1}||_F + eps)` on the union edge set
3. **Normalized structural Hamming distance** between consecutive graphs

Each signal is standardized using robust (median / MAD) statistics from a
warmup period, then combined with equal weights by default.

### Drift Detection

We monitor the fused drift score with a **double-sided CUSUM**:

```
C_pos = max(0, C_pos + d_t - k)
C_neg = max(0, C_neg - d_t - k)
if C_pos > h or C_neg > h: candidate
```

The threshold is adaptive: `h = mu_d + kappa * sigma_d`, with robust
statistics from the warmup period. A candidate is confirmed by a
permutation test on residuals (200 permutations by default).

### Incremental Update

Upon confirmed drift:

1. Locate changed variables via per-variable drift scores.
2. Expand each seed to its directed Markov blanket.
3. Re-run DirectLiNGAM only on the local subset, warm-started with the
   previous causal order.
4. Merge the local subgraph back into the global graph.
5. Run consistency checks (acyclicity, SHD jump, residual variance,
   coefficient explosion). If any fails, fall back to full retraining.

## Experiments

### Datasets

| Dataset | Variables | Samples | Use |
|---------|-----------|---------|-----|
| Synthetic (drifting) | 8 | 5000 | Primary evaluation |
| Tennessee-Eastman | 52 | 960 | Qualitative validation |
| SKAB | 8 | 1140 | Drift detection case study |

### Drift Scenarios

| ID | Drift type | Mode | Description |
|----|------------|------|-------------|
| S1 | structure | abrupt | Edge add/remove at window 5 |
| S2 | coefficients | abrupt | Coefficient scaling + perturbation |
| S3 | both | abrupt | Structure + coefficients at window 5 |
| S4 | structure | gradual | Linear interpolation over windows 3–7 |
| S5 | coefficients | gradual | Linear interpolation over windows 3–7 |

### Baselines

| ID | Name | Description |
|----|------|-------------|
| B1 | Full retrain every window | Quality upper bound |
| B2 | Detect + full retrain | Isolates detection contribution |
| B3 | Sliding window, no update | Lower bound |
| B4 | Ours (incremental) | Detection + MB local re-learn + fallback |

### Main results

Averaged over 5 scenarios × 10 seeds.

| Method | SHD pre | SHD post | F1 post | Coef MSE | Det F1 | Update/retrain |
|--------|---------|----------|---------|----------|--------|----------------|
| B1 (Full retrain) | 9.57 | 12.13 | 0.742 | 0.133 | — | 1.0 |
| B2 (Detect + full) | 9.57 | 12.13 | 0.742 | 0.133 | 0.632 | 1.0 |
| B3 (Frozen) | 9.48 | 12.46 | 0.729 | 0.147 | — | — |
| **B4 (Ours)** | **9.58** | **12.06** | **0.743** | **0.133** | **0.634** | **71.7×** |

**B4 matches or slightly beats the full-retrain baseline on graph
recovery (SHD, F1, coefficient MSE) while running 71.7× faster on drift
windows.** Fallback rate is 2.2%.

## Ablations

### Signal ablation (A1–A6)

Detection performance by signal combination. All modes use the same
detector, CUSUM, and update pipeline; only the fused drift score
changes.

| Mode | Signals | Precision | Recall | F1 | Confirmed windows |
|------|---------|-----------|--------|----|-------------------|
| A1 | ED only | 0.36 | 0.38 | 0.36 | 65 |
| A2 | ΔB only | **0.74** | **0.82** | **0.76** | 71 |
| A3 | SHD only | 0.61 | 0.68 | 0.62 | 95 |
| A4 | ED + ΔB | 0.61 | 0.69 | 0.63 | 115 |
| A5 | ED + ΔB + SHD | 0.66 | 0.71 | 0.67 | 158 |
| A6 | ΔB + SHD | 0.72 | 0.79 | 0.73 | 116 |

Per-scenario detection recall:

| Scenario | A1 | A2 | A3 | A4 | A5 | A6 |
|----------|----|----|----|----|----|----|
| S1 (structure / abrupt) | 0.70 | 1.00 | 1.00 | 0.90 | 1.00 | 1.00 |
| S2 (coef / abrupt) | 0.10 | 0.80 | 0.40 | 0.50 | 0.50 | 0.60 |
| S3 (both / abrupt) | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S4 (structure / gradual) | 0.40 | 0.90 | 0.75 | 0.75 | 0.80 | 0.85 |
| S5 (coef / gradual) | 0.10 | 0.40 | 0.25 | 0.30 | 0.25 | 0.50 |

**Findings.**

1. **ΔB alone (A2) gives the highest overall detection F1 (0.76)** and
   dominates on coefficient-only drift (S2, S5).
2. **SHD (A3) is complementary but weaker**, nearly blind to
   coefficient drift (S2 recall 0.40, S5 recall 0.25).
3. **Residual energy distance (ED) is a net negative under max fusion.**
   Adding ED to ΔB drops F1 from 0.76 (A2) to 0.63 (A4). ED has a heavy
   right tail in quiet windows, which inflates the adaptive CUSUM
   threshold and suppresses genuine ΔB peaks.
4. **The fused score affects detection, not graph recovery.** SHD
   post-drift is 12.1–12.3 across all six modes, confirming that once
   drift is confirmed, the fallback-protected update converges to a
   comparable graph regardless of which signal triggered it.

**Recommendation:** use A2 (ΔB only) or A6 (ΔB + SHD) for the default
signal set, not the three-signal A5.

### Update scope ablation (U1–U4)

Detection and graph quality by update scope. All modes use the same
detector and fallback; only the local re-learning subset changes.

| Mode | Scope | Detection F1 | SHD post | F1 post | Update / retrain | Fallback |
|------|-------|--------------|----------|---------|------------------|----------|
| U1 | Seeds only | 0.655 | 12.248 | 0.7423 | 63.1× | 0.012 |
| U2 | Seeds + parents | 0.655 | 12.252 | 0.7422 | 62.3× | 0.012 |
| U3 | Seeds + MB (default) | **0.668** | 12.248 | **0.7424** | 60.6× | 0.012 |
| U4 | Full retrain | 0.631 | **12.128** | 0.7418 | — | 1.000 |

SHD gap relative to U4 (positive = worse than full retrain):

| Scenario | U1 | U2 | U3 |
|----------|----|----|----|
| S1 | 0.20 | 0.20 | **0.16** |
| S2 | 0.06 | 0.06 | 0.06 |
| S3 | **0.12** | 0.14 | 0.16 |
| S4 | 0.24 | 0.24 | 0.24 |
| S5 | −0.02 | −0.02 | −0.02 |

**Findings.**

1. **All three local scopes are equivalent in cost and graph quality.**
   U1, U2, and U3 run 60–63× faster than full retraining on drift
   windows and produce SHD within 0.01 of each other. The
   consistency-checked fallback absorbs most of the difference between
   update scopes at this data scale.
2. **No single scope dominates.** U3 has the smallest SHD gap on S1, U1
   on S3, and all three are indistinguishable on S2, S4, S5.
3. **U3 gives a marginal detection benefit.** Detection F1 is 0.668 for
   U3 versus 0.655 for U1/U2 and 0.631 for U4.
4. **U4 has the best graph quality but no incremental path.** Full
   retraining achieves SHD 12.128 but its update/retrain ratio is
   undefined.
5. **S4 remains the hard case.** All three local scopes fall 0.24 SHD
   behind U4 on gradual structure drift. This is a data-scale
   limitation.

**Recommendation:** keep U3 (Markov-blanket expansion) as default for
its marginal detection F1 advantage, but do **not** claim it dominates
U1 or U2.

## Limitations

1. **Data scale.** At 8 variables with 500 samples per window, the SHD
   noise floor (≈ 10) is comparable to the drift signal. For gradual
   structure drift (S4), the frozen baseline matches or slightly beats
   full retraining. Increasing window size or variable count should
   separate the methods; this is deferred to future work.
2. **Gradual coefficient drift (S5).** Even with the best signal
   combination, detection recall is ≈ 0.5. Coefficient change accumulates
   slowly, and CUSUM must trade off delay against false positives.
3. **MLX backend.** MLX is used for the hot numerical routines (cumulants,
   least squares, matrix products). Small-matrix operations fall back to
   NumPy. The boundary between the two is benchmarked, not assumed.

## Reproducing Results

```bash
# Full reproduction (~3 hours on M3 Pro 18GB)
python -m experiments.run_main --out results
python -m experiments.run_ablation_signals --out results/ablation_signals \
    --modes A1 A2 A3 A4 A5 A6
python -m experiments.aggregate_ablation \
    --main results/ablation_signals/ablation_signals.csv \
    --out results/ablation_signals
python -m experiments.run_ablation_update --out results/ablation_update
python -m experiments.aggregate_ablation \
    --main results/ablation_update/ablation_update.csv \
    --out results/ablation_update
python -m experiments.run_cusum_sweep --out results
python -m experiments.plots --main results/main.csv \
    --out results/figures --root results
```

Or interactively via `notebooks/experiment.ipynb`.

```bash
# Unit tests
python -m pytest tests/
```

## Project Structure

```
mlx-incremental-causal-discovery/
├── online_causal.py        # Main OnlineCausalDiscovery orchestration
├── lingam_mlx.py           # MLX DirectLiNGAM engine
├── drift_detector.py       # Multi-signal drift detection
├── incremental.py          # MB local re-learning + fallback
├── graph_ops.py            # Graph operations (MB, SHD, merge, checks)
├── stats.py                # Energy distance, coefficient change, tests
├── visualize.py            # Causal graph animation
├── data/
│   ├── synthetic.py        # Drifting synthetic data generator
│   └── loaders.py          # Tennessee-Eastman, SKAB loaders
├── experiments/
│   ├── config.py           # Experiment configuration
│   ├── run_main.py         # Main experiment
│   ├── run_ablation_signals.py
│   ├── run_ablation_update.py
│   ├── run_cusum_sweep.py
│   ├── aggregate_ablation.py
│   └── plots.py            # Figures for main + ablation results
├── notebooks/
│   └── experiment.ipynb    # Full reproduction walkthrough
├── tests/
│   ├── test_stats.py
│   ├── test_graph_ops.py
│   └── test_lingam.py
├── requirements.txt
└── README.md
```

## References

- Shimizu et al., "A Linear Non-Gaussian Acyclic Model for Causal
  Discovery", JMLR 2006.
- Page, "Continuous Inspection Schemes", Biometrika 1954 (CUSUM).
- MLX: https://github.com/ml-explore/mlx
- lingam package: https://github.com/cdt15/lingam
