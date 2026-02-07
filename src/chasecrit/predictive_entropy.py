from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import csv_write, ensure_dir
from .plotting import plot_heatmap, plot_metric_vs_w_align, plot_scatter
from .report import load_results_csv


@dataclass(frozen=True)
class GroupStats:
    n: int
    mean: float
    ci95: float


def _group_stats(values: list[float]) -> GroupStats:
    n = len(values)
    if n == 0:
        return GroupStats(n=0, mean=float("nan"), ci95=float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    return GroupStats(n=n, mean=mean, ci95=float(1.96 * sem) if n > 1 else 0.0)


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    mx = float(xx.mean())
    my = float(yy.mean())
    dx = xx - mx
    dy = yy - my
    den = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if den <= 0.0:
        return float("nan")
    return float(np.sum(dx * dy) / den)


def _read_probe_heading(path: Path) -> list[float]:
    vals: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "probe_heading_deg" not in (reader.fieldnames or []):
            return vals
        for row in reader:
            raw = row.get("probe_heading_deg", "")
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except Exception:
                continue
            if math.isnan(v):
                continue
            vals.append(v)
    return vals


def _ema_circular_deg(angles_deg: list[float], span: int) -> np.ndarray:
    if not angles_deg:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(angles_deg, dtype=np.float64)
    rad = np.deg2rad(arr)
    x = np.cos(rad)
    y = np.sin(rad)
    alpha = 2.0 / (float(span) + 1.0)
    ex = np.empty_like(x)
    ey = np.empty_like(y)
    ex[0] = x[0]
    ey[0] = y[0]
    for i in range(1, x.shape[0]):
        ex[i] = alpha * x[i] + (1.0 - alpha) * ex[i - 1]
        ey[i] = alpha * y[i] + (1.0 - alpha) * ey[i - 1]
    out = (np.degrees(np.arctan2(ey, ex)) + 360.0) % 360.0
    return out.astype(np.float64)


def _discretize_deg(angles_deg: np.ndarray, bins: int) -> np.ndarray:
    if angles_deg.size == 0:
        return np.zeros((0,), dtype=np.int32)
    scaled = np.around((angles_deg / 360.0) * float(bins))
    return np.mod(scaled.astype(np.int64), bins).astype(np.int32)


def _build_ngram_model(token_seqs: list[np.ndarray], order: int, bins: int) -> dict[tuple[int, ...], np.ndarray]:
    counts: dict[tuple[int, ...], np.ndarray] = {}
    for seq in token_seqs:
        if seq.shape[0] <= order:
            continue
        for t in range(order, seq.shape[0]):
            ctx = tuple(int(x) for x in seq[t - order : t].tolist())
            nxt = int(seq[t])
            if ctx not in counts:
                counts[ctx] = np.zeros((bins,), dtype=np.float64)
            counts[ctx][nxt] += 1.0
    return counts


def _entropy_from_probs(probs: np.ndarray) -> float:
    p = probs[probs > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _sequence_entropy_stats(
    tokens: np.ndarray,
    model: dict[tuple[int, ...], np.ndarray],
    *,
    order: int,
    bins: int,
    smoothing: float = 0.5,
) -> tuple[float, float, int]:
    if tokens.shape[0] <= order + 1:
        return float("nan"), float("nan"), 0
    ent: list[float] = []
    base = smoothing * float(bins)
    uniform = np.full((bins,), 1.0 / float(bins), dtype=np.float64)
    for t in range(order, tokens.shape[0]):
        ctx = tuple(int(x) for x in tokens[t - order : t].tolist())
        raw = model.get(ctx)
        if raw is None:
            probs = uniform
        else:
            probs = (raw + smoothing) / (float(raw.sum()) + base)
        ent.append(_entropy_from_probs(probs))
    if len(ent) < 2:
        return float("nan"), float("nan"), len(ent)
    arr = np.asarray(ent, dtype=np.float64)
    var_h = float(np.var(arr))
    iqr_h = float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))
    return var_h, iqr_h, int(arr.shape[0])


def run_predictive_entropy_report(
    *,
    sweep_dir: str | Path,
    out_dir: str | Path,
    title: str,
    bins: int = 72,
    ema_span: int = 10,
    order: int = 2,
) -> Path:
    sweep = Path(sweep_dir)
    out = ensure_dir(out_dir)
    figs = ensure_dir(out / "figs")

    rows = load_results_csv(sweep / "results.csv")

    run_rows: list[dict[str, Any]] = []
    token_bank: list[np.ndarray] = []
    for r in rows:
        run_dir_raw = str(r.get("run_dir", "")).strip()
        if run_dir_raw == "":
            continue
        ts_path = Path(run_dir_raw) / "timeseries.csv"
        if not ts_path.exists():
            continue
        headings = _read_probe_heading(ts_path)
        if len(headings) < (order + 2):
            continue
        smooth = _ema_circular_deg(headings, span=ema_span)
        toks = _discretize_deg(smooth, bins=bins)
        if toks.shape[0] < (order + 2):
            continue
        token_bank.append(toks)
        run_rows.append(
            {
                "speed_ratio": float(r["speed_ratio"]),
                "w_align": float(r["w_align"]),
                "seed": int(r["seed"]),
                "safe_frac": float(r["safe_frac"]),
                "captured_frac": float(r["captured_frac"]),
                "chi": float(r["chi"]),
                "run_dir": run_dir_raw,
                "_tokens": toks,
            }
        )

    if not run_rows:
        raise RuntimeError("No valid per-run timeseries with probe_heading_deg found. Use a sweep generated with --save-runs and save_timeseries.")

    model = _build_ngram_model(token_bank, order=order, bins=bins)

    per_run_rows: list[dict[str, Any]] = []
    for r in run_rows:
        var_h, iqr_h, n_eff = _sequence_entropy_stats(r["_tokens"], model, order=order, bins=bins)
        out_row = {k: v for k, v in r.items() if k != "_tokens"}
        out_row["pred_entropy_var"] = var_h
        out_row["pred_entropy_iqr"] = iqr_h
        out_row["pred_entropy_n"] = n_eff
        per_run_rows.append(out_row)

    csv_write(
        out / "per_run_pred_entropy.csv",
        per_run_rows,
        fieldnames=[
            "speed_ratio",
            "w_align",
            "seed",
            "safe_frac",
            "captured_frac",
            "chi",
            "pred_entropy_var",
            "pred_entropy_iqr",
            "pred_entropy_n",
            "run_dir",
        ],
    )

    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in per_run_rows:
        grouped.setdefault((float(r["speed_ratio"]), float(r["w_align"])), []).append(r)

    grouped_rows: list[dict[str, Any]] = []
    for (sr, wa), items in sorted(grouped.items()):
        s_safe = _group_stats([float(x["safe_frac"]) for x in items])
        s_varh = _group_stats([float(x["pred_entropy_var"]) for x in items if not math.isnan(float(x["pred_entropy_var"]))])
        s_iqrh = _group_stats([float(x["pred_entropy_iqr"]) for x in items if not math.isnan(float(x["pred_entropy_iqr"]))])
        grouped_rows.append(
            {
                "speed_ratio": sr,
                "w_align": wa,
                "n": s_safe.n,
                "safe_mean": s_safe.mean,
                "safe_ci95": s_safe.ci95,
                "pred_entropy_var_mean": s_varh.mean,
                "pred_entropy_var_ci95": s_varh.ci95,
                "pred_entropy_iqr_mean": s_iqrh.mean,
                "pred_entropy_iqr_ci95": s_iqrh.ci95,
            }
        )

    csv_write(
        out / "group_summary_pred_entropy.csv",
        grouped_rows,
        fieldnames=[
            "speed_ratio",
            "w_align",
            "n",
            "safe_mean",
            "safe_ci95",
            "pred_entropy_var_mean",
            "pred_entropy_var_ci95",
            "pred_entropy_iqr_mean",
            "pred_entropy_iqr_ci95",
        ],
    )

    plot_metric_vs_w_align(
        grouped_rows=grouped_rows,
        metric_mean="pred_entropy_var_mean",
        metric_ci="pred_entropy_var_ci95",
        out_path=figs / "pred_entropy_var_vs_w_align.png",
        title="Predictive entropy variance vs w_align (mean ± 95% CI)",
        ylabel="Var(H_pred)",
    )
    plot_metric_vs_w_align(
        grouped_rows=grouped_rows,
        metric_mean="safe_mean",
        metric_ci="safe_ci95",
        out_path=figs / "safe_vs_w_align.png",
        title="safe_frac vs w_align (mean ± 95% CI)",
        ylabel="safe_frac",
    )
    plot_scatter(
        grouped_rows=grouped_rows,
        x_key="pred_entropy_var_mean",
        y_key="safe_mean",
        out_path=figs / "scatter_safe_vs_pred_entropy_var.png",
        title="safe_frac vs predictive entropy variance (group means)",
        xlabel="Var(H_pred) mean",
        ylabel="safe_frac mean",
    )
    plot_heatmap(
        grouped_rows=grouped_rows,
        value_key="pred_entropy_var_mean",
        out_path=figs / "heatmap_pred_entropy_var_mean.png",
        title="Heatmap of predictive entropy variance mean",
    )

    by_sr: dict[float, list[dict[str, Any]]] = {}
    for r in grouped_rows:
        by_sr.setdefault(float(r["speed_ratio"]), []).append(r)

    md: list[str] = []
    md.append(f"# {title}\n")
    md.append("## Setup\n")
    md.append(f"- Sweep directory: `{sweep.as_posix()}`\n")
    md.append(f"- Bins: `{bins}`\n")
    md.append(f"- EMA span: `{ema_span}`\n")
    md.append(f"- N-gram order: `{order}`\n")
    md.append(f"- Valid runs used: `{len(per_run_rows)}`\n")
    md.append("\n## Per-speed correlations (group means)\n")
    md.append("| v_p/v_e | corr(safe, Var(H_pred)) |\n")
    md.append("|---:|---:|\n")
    for sr in sorted(by_sr):
        g = by_sr[sr]
        corr_val = _corr(
            [float(x["pred_entropy_var_mean"]) for x in g],
            [float(x["safe_mean"]) for x in g],
        )
        md.append(f"| {sr:g} | {corr_val:.3f} |\n")

    pooled_corr = _corr(
        [float(x["pred_entropy_var_mean"]) for x in grouped_rows],
        [float(x["safe_mean"]) for x in grouped_rows],
    )
    md.append(f"\n- pooled corr(safe, Var(H_pred)) across all grouped points: `{pooled_corr:.3f}`\n")

    md.append("\n## Plots\n")
    md.append("![pred_entropy_var_vs_w_align](figs/pred_entropy_var_vs_w_align.png)\n")
    md.append("![safe_vs_w_align](figs/safe_vs_w_align.png)\n")
    md.append("![scatter_safe_vs_pred_entropy_var](figs/scatter_safe_vs_pred_entropy_var.png)\n")
    md.append("![heatmap_pred_entropy_var_mean](figs/heatmap_pred_entropy_var_mean.png)\n")

    report_path = out / "report_pred_entropy.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    return report_path
