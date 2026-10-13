"""
visualize.py
------------
Visualization utilities: causal graph animation, drift score plots,
SHD time series, delay-FPR curves.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


# ---------------------------------------------------------------------------
# Graph drawing
# ---------------------------------------------------------------------------

def draw_causal_graph(
    adj: np.ndarray,
    labels: list[str],
    ax: plt.Axes,
    title: str = "",
    drift_point: bool = False,
    layout: Optional[dict] = None,
) -> dict:
    """Draw a causal DAG on *ax*.  Returns the node position dict."""
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(adj != 0)
    for i, j in zip(rows, cols):
        G.add_edge(int(j), int(i))  # adj[i,j] means j → i

    if layout is None:
        try:
            layout = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            layout = nx.spring_layout(G, seed=42)

    node_color = "#ff6b6b" if drift_point else "#4ecdc4"
    nx.draw_networkx(
        G,
        pos=layout,
        labels={i: labels[i] for i in range(n)},
        ax=ax,
        node_color=node_color,
        node_size=600,
        font_size=7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return layout


# ---------------------------------------------------------------------------
# Animated GIF
# ---------------------------------------------------------------------------

def animate_causal_evolution(
    adj_matrices: list[np.ndarray],
    labels: list[str],
    drift_points: Optional[list[int]] = None,
    save_path: str = "causal_evolution.gif",
    fps: int = 2,
    figsize: tuple[float, float] = (5, 4),
) -> None:
    """Create an animated GIF of the causal graph evolving over windows.

    Args:
        adj_matrices: List of (n_nodes, n_nodes) adjacency matrices per window.
        labels:       Node label strings.
        drift_points: Window indices where drift was confirmed.
        save_path:    Output GIF path.
        fps:          Frames per second.
    """
    drift_set = set(drift_points or [])
    frames: list[np.ndarray] = []
    layout: Optional[dict] = None

    for w, adj in enumerate(adj_matrices):
        fig, ax = plt.subplots(figsize=figsize)
        is_drift = w in drift_set
        layout = draw_causal_graph(
            adj,
            labels=labels,
            ax=ax,
            title=f"Window {w}" + (" [DRIFT]" if is_drift else ""),
            drift_point=is_drift,
            layout=layout,
        )
        fig.tight_layout()

        # Render to RGB array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)
        plt.close(fig)

    duration = 1.0 / fps
    imageio.mimsave(save_path, frames, duration=duration, loop=0)
    print(f"Saved causal evolution GIF → {save_path}")


# ---------------------------------------------------------------------------
# Drift score plot
# ---------------------------------------------------------------------------

def plot_drift_scores(
    drift_scores: list[float],
    drift_points: Optional[list[int]] = None,
    true_drift_windows: Optional[list[int]] = None,
    cusum_pos: Optional[list[float]] = None,
    cusum_neg: Optional[list[float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot fused drift score over windows with annotated drift events."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1, ax2 = axes

    scores = np.array([s if not np.isnan(s) else 0.0 for s in drift_scores])
    windows = np.arange(len(scores))

    ax1.plot(windows, scores, color="#2980b9", label="Fused drift score", linewidth=1.5)

    if true_drift_windows:
        for tw in true_drift_windows:
            ax1.axvline(tw, color="green", linestyle="--", alpha=0.7, label="True drift")

    if drift_points:
        for dp in drift_points:
            ax1.axvline(dp, color="red", linestyle=":", alpha=0.8, label="Detected drift")

    handles, lbls = ax1.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax1.set_ylabel("Drift score (z)")
    ax1.set_title("Fused Drift Score")
    ax1.grid(True, alpha=0.3)

    if cusum_pos is not None and cusum_neg is not None:
        ax2.plot(windows, cusum_pos, color="#e74c3c", label="CUSUM+", linewidth=1.2)
        ax2.plot(windows, cusum_neg, color="#8e44ad", label="CUSUM−", linewidth=1.2)
        ax2.legend(fontsize=8)
        ax2.set_ylabel("CUSUM stat")
        ax2.set_title("Double-sided CUSUM")
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Window index")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved drift score plot → {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# SHD time series
# ---------------------------------------------------------------------------

def plot_shd_series(
    adj_matrices: list[np.ndarray],
    true_adjs: Optional[list[np.ndarray]] = None,
    drift_points: Optional[list[int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot SHD between consecutive estimated graphs (and vs. ground truth)."""
    from graph_ops import structural_hamming_distance

    fig, ax = plt.subplots(figsize=(10, 4))
    n = len(adj_matrices)

    # Consecutive SHD
    consec_shd = [
        structural_hamming_distance(adj_matrices[i - 1], adj_matrices[i])
        for i in range(1, n)
    ]
    ax.plot(range(1, n), consec_shd, label="SHD (consecutive)", color="#2980b9", linewidth=1.5)

    if true_adjs is not None and len(true_adjs) == n:
        gt_shd = [
            structural_hamming_distance(true_adjs[i], adj_matrices[i])
            for i in range(n)
        ]
        ax.plot(range(n), gt_shd, label="SHD (vs ground truth)", color="#27ae60",
                linewidth=1.5, linestyle="--")

    if drift_points:
        for dp in drift_points:
            ax.axvline(dp, color="red", linestyle=":", alpha=0.7)

    ax.set_xlabel("Window index")
    ax.set_ylabel("SHD")
    ax.set_title("Structural Hamming Distance Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved SHD plot → {save_path}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Delay-FPR curve
# ---------------------------------------------------------------------------

def plot_delay_fpr_curve(
    results_by_scenario: dict[str, dict],
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot detection delay vs. false-positive rate for multiple scenarios.

    Each entry in *results_by_scenario* is keyed by scenario name and has
    keys 'delay' (int or float) and 'fpr' (float in [0, 1]).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, metrics in results_by_scenario.items():
        ax.scatter(metrics["fpr"], metrics["delay"], label=name, s=80)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Detection Delay (windows)")
    ax.set_title("Detection Delay vs. False Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved delay-FPR plot → {save_path}")
    if show:
        plt.show()
    return fig
