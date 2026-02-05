from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupStats:
    n: int
    mean: float
    std: float
    sem: float
    ci95: float


def _group_stats(values: list[float]) -> GroupStats:
    n = len(values)
    if n == 0:
        return GroupStats(n=0, mean=float("nan"), std=float("nan"), sem=float("nan"), ci95=float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    ci95 = float(1.96 * sem) if n > 1 else 0.0
    return GroupStats(n=n, mean=mean, std=std, sem=sem, ci95=ci95)


def load_results_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out: dict[str, Any] = dict(row)
            for k in ("w_align", "speed_ratio", "angle_noise", "safe_frac", "captured_frac", "P_mean", "P_var", "chi"):
                if k in out and out[k] != "" and out[k] is not None:
                    out[k] = float(out[k])
            for k in ("seed", "alive", "safe", "captured", "steps_recorded"):
                if k in out and out[k] != "" and out[k] is not None:
                    out[k] = int(float(out[k]))
            rows.append(out)
    return rows


def summarize_by_group(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, GroupStats]]:
    by: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in rows:
        key = (float(r["speed_ratio"]), float(r["w_align"]))
        by.setdefault(key, []).append(r)

    out: dict[tuple[float, float], dict[str, GroupStats]] = {}
    for key, items in by.items():
        out[key] = {
            "safe_frac": _group_stats([float(x["safe_frac"]) for x in items]),
            "captured_frac": _group_stats([float(x["captured_frac"]) for x in items]),
            "P_mean": _group_stats([float(x["P_mean"]) for x in items]),
            "P_var": _group_stats([float(x["P_var"]) for x in items]),
            "chi": _group_stats([float(x.get("chi", float("nan"))) for x in items]),
        }
    return out


def summarize_by_two_keys(
    rows: list[dict[str, Any]],
    *,
    key_a: str,
    key_b: str,
) -> dict[tuple[float, float], dict[str, GroupStats]]:
    by: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for r in rows:
        a = float(r[key_a])
        b = float(r[key_b])
        by.setdefault((a, b), []).append(r)

    out: dict[tuple[float, float], dict[str, GroupStats]] = {}
    for key, items in by.items():
        out[key] = {
            "safe_frac": _group_stats([float(x["safe_frac"]) for x in items]),
            "captured_frac": _group_stats([float(x["captured_frac"]) for x in items]),
            "P_mean": _group_stats([float(x["P_mean"]) for x in items]),
            "P_var": _group_stats([float(x["P_var"]) for x in items]),
            "chi": _group_stats([float(x.get("chi", float("nan"))) for x in items]),
        }
    return out


def write_group_summary_csv(
    out_path: str | Path,
    grouped: dict[tuple[float, float], dict[str, GroupStats]],
) -> None:
    p = Path(out_path)
    keys = sorted(grouped.keys())
    fieldnames = [
        "speed_ratio",
        "w_align",
        "n",
        "safe_mean",
        "safe_ci95",
        "captured_mean",
        "captured_ci95",
        "P_mean",
        "P_mean_ci95",
        "P_var",
        "P_var_ci95",
        "chi_mean",
        "chi_ci95",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (sr, wa) in keys:
            g = grouped[(sr, wa)]
            row = {
                "speed_ratio": sr,
                "w_align": wa,
                "n": g["safe_frac"].n,
                "safe_mean": g["safe_frac"].mean,
                "safe_ci95": g["safe_frac"].ci95,
                "captured_mean": g["captured_frac"].mean,
                "captured_ci95": g["captured_frac"].ci95,
                "P_mean": g["P_mean"].mean,
                "P_mean_ci95": g["P_mean"].ci95,
                "P_var": g["P_var"].mean,
                "P_var_ci95": g["P_var"].ci95,
                "chi_mean": g["chi"].mean,
                "chi_ci95": g["chi"].ci95,
            }
            w.writerow(row)
