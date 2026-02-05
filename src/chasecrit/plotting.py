from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_vs_w_align(
    *,
    grouped_rows: list[dict[str, Any]],
    metric_mean: str,
    metric_ci: str,
    out_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """
    grouped_rows: rows with columns: speed_ratio, w_align, safe_mean, safe_ci95, etc.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed_ratios = sorted({float(r["speed_ratio"]) for r in grouped_rows})
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)

    for sr in speed_ratios:
        rows = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
        rows.sort(key=lambda r: float(r["w_align"]))
        x = np.array([float(r["w_align"]) for r in rows])
        y = np.array([float(r[metric_mean]) for r in rows])
        yerr = np.array([float(r[metric_ci]) for r in rows])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=f"v_p/v_e={sr:g}")

    ax.set_title(title)
    ax.set_xlabel("w_align")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_vs_x(
    *,
    grouped_rows: list[dict[str, Any]],
    x_key: str,
    metric_mean: str,
    metric_ci: str,
    out_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed_ratios = sorted({float(r["speed_ratio"]) for r in grouped_rows})
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)

    for sr in speed_ratios:
        rows = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
        rows.sort(key=lambda r: float(r[x_key]))
        x = np.array([float(r[x_key]) for r in rows])
        y = np.array([float(r[metric_mean]) for r in rows])
        yerr = np.array([float(r[metric_ci]) for r in rows])
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=f"v_p/v_e={sr:g}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(
    *,
    grouped_rows: list[dict[str, Any]],
    value_key: str,
    out_path: str | Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed_ratios = sorted({float(r["speed_ratio"]) for r in grouped_rows})
    w_values = sorted({float(r["w_align"]) for r in grouped_rows})

    grid = np.full((len(speed_ratios), len(w_values)), np.nan, dtype=np.float64)
    index = {(float(r["speed_ratio"]), float(r["w_align"])): r for r in grouped_rows}
    for i, sr in enumerate(speed_ratios):
        for j, w in enumerate(w_values):
            r = index.get((sr, w))
            if r is not None:
                grid[i, j] = float(r[value_key])

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("w_align")
    ax.set_ylabel("v_p/v_e")
    ax.set_xticks(np.arange(len(w_values)))
    ax.set_xticklabels([f"{w:g}" for w in w_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(speed_ratios)))
    ax.set_yticklabels([f"{sr:g}" for sr in speed_ratios])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_key)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter(
    *,
    grouped_rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    out_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speed_ratios = sorted({float(r["speed_ratio"]) for r in grouped_rows})
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)

    for sr in speed_ratios:
        rows = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
        x = np.array([float(r[x_key]) for r in rows])
        y = np.array([float(r[y_key]) for r in rows])
        ax.scatter(x, y, s=28, alpha=0.85, label=f"v_p/v_e={sr:g}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
