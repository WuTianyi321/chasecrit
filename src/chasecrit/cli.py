from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .config import (
    EvaderConfig,
    ExperimentConfig,
    PursuerConfig,
    RunConfig,
    SafeZoneConfig,
    WorldConfig,
    load_config,
)
from .bench import run_small_benchmark
from .io_utils import csv_write, ensure_dir, json_dump
from .predictive_entropy import run_predictive_entropy_report
from .report import load_results_csv, summarize_by_group, summarize_by_two_keys, write_group_summary_csv
from .plotting import plot_heatmap, plot_metric_vs_w_align, plot_metric_vs_x, plot_scatter
from .sim import run_once, run_summary


def _base_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="Path to .json or .toml config file")
    p.add_argument("--out", type=str, default=None, help="Override output directory")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chasecrit", description="ChaseCrit: near-critical swarms in pursuit-evasion tasks")
    sub = p.add_subparsers(dest="cmd", required=False)

    run = sub.add_parser("run", help="Run a single episode (default)")
    _base_args(run)
    run.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    run.add_argument("--steps", type=int, default=None, help="Override number of steps")

    sweep = sub.add_parser("sweep-w-align", help="Sweep evader w_align over values and seeds")
    _base_args(sweep)
    sweep.add_argument("--values", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="w_align values")
    sweep.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds")
    sweep.add_argument("--steps", type=int, default=None, help="Override number of steps")
    sweep.add_argument("--no-save-timeseries", action="store_true", help="Do not save per-run timeseries")
    sweep.add_argument("--no-save-events", action="store_true", help="Do not save per-run events")

    grid = sub.add_parser("sweep-grid", help="Sweep w_align and pursuer speed_ratio over grids (fixed pursuer count)")
    _base_args(grid)
    grid.add_argument("--w-align", type=float, nargs="+", required=True, help="w_align values")
    grid.add_argument("--speed-ratio", type=float, nargs="+", required=True, help="v_p/v_e values")
    grid.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds")
    grid.add_argument("--steps", type=int, default=None, help="Override number of steps")
    grid.add_argument("--no-save-timeseries", action="store_true", help="Do not save per-run timeseries")
    grid.add_argument("--no-save-events", action="store_true", help="Do not save per-run events")
    grid.add_argument("--save-runs", action="store_true", help="Save per-run directories under the sweep dir (slower)")
    grid.add_argument("--jobs", type=int, default=1, help="Parallel workers (processes). Default 1.")
    grid.add_argument("--progress-file", type=str, default="progress.jsonl", help="Progress log filename inside sweep dir")

    rep = sub.add_parser("report", help="Create plots and a markdown report from a sweep directory")
    rep.add_argument("--sweep-dir", type=str, required=True, help="Sweep directory containing results.csv")
    rep.add_argument("--out-dir", type=str, required=True, help="Output directory for report and figures")
    rep.add_argument("--title", type=str, default="Fixed pursuer-count scan", help="Report title")

    phase = sub.add_parser("phase-sweep-noise", help="Sweep angle_noise in a phase-identification setting (no pursuers, no safe zones)")
    _base_args(phase)
    phase.add_argument("--noise", type=float, nargs="+", required=True, help="angle_noise values (radians)")
    phase.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds")
    phase.add_argument("--steps", type=int, default=None, help="Override number of steps")
    phase.add_argument("--jobs", type=int, default=1, help="Parallel workers (processes)")
    phase.add_argument("--progress-file", type=str, default="progress.jsonl", help="Progress log filename inside sweep dir")

    noise_rep = sub.add_parser("report-noise", help="Create plots + markdown report for a noise sweep (speed_ratio × angle_noise)")
    noise_rep.add_argument("--sweep-dir", type=str, required=True, help="Sweep directory containing results.csv")
    noise_rep.add_argument("--out-dir", type=str, required=True, help="Output directory for report and figures")
    noise_rep.add_argument("--title", type=str, default="Noise sweep report", help="Report title")

    pred_rep = sub.add_parser(
        "report-pred-entropy",
        help="Compute predictive-entropy-variance metrics from per-run probe trajectories",
    )
    pred_rep.add_argument("--sweep-dir", type=str, required=True, help="Sweep directory containing results.csv and per-run timeseries")
    pred_rep.add_argument("--out-dir", type=str, required=True, help="Output directory for predictive-entropy report")
    pred_rep.add_argument("--title", type=str, default="Predictive entropy variance report", help="Report title")
    pred_rep.add_argument("--bins", type=int, default=72, help="Discretization bins for heading angle")
    pred_rep.add_argument("--ema-span", type=int, default=10, help="Circular EMA span for heading smoothing")
    pred_rep.add_argument("--order", type=int, default=2, help="N-gram order for predictor context")

    sweep_noise = sub.add_parser("sweep-noise", help="Sweep angle_noise and pursuer speed_ratio (fixed pursuer count)")
    _base_args(sweep_noise)
    sweep_noise.add_argument("--noise", type=float, nargs="+", required=True, help="angle_noise values (radians)")
    sweep_noise.add_argument("--speed-ratio", type=float, nargs="+", required=True, help="v_p/v_e values")
    sweep_noise.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds")
    sweep_noise.add_argument("--steps", type=int, default=None, help="Override number of steps")
    sweep_noise.add_argument("--w-align", type=float, default=None, help="Override w_align (optional)")
    sweep_noise.add_argument("--no-save-timeseries", action="store_true", help="Do not save per-run timeseries")
    sweep_noise.add_argument("--no-save-events", action="store_true", help="Do not save per-run events")
    sweep_noise.add_argument("--save-runs", action="store_true", help="Save per-run directories under the sweep dir (slower)")
    sweep_noise.add_argument("--jobs", type=int, default=1, help="Parallel workers (processes). Default 1.")
    sweep_noise.add_argument("--progress-file", type=str, default="progress.jsonl", help="Progress log filename inside sweep dir")

    bench = sub.add_parser("bench-speed", help="Small benchmark for speed and behavior parity (NumPy vs Numba)")
    _base_args(bench)
    bench.add_argument("--iterations", type=int, default=300, help="Iterations for evader_step micro benchmark")
    bench.add_argument("--steps", type=int, default=300, help="Episode steps for simulation benchmark")
    bench.add_argument("--repeats", type=int, default=3, help="Number of repeated measurements")
    bench.add_argument("--seed", type=int, default=0, help="Base random seed")
    bench.add_argument("--json-out", type=str, default=None, help="Optional path to save benchmark JSON")

    return p


def _write_progress(
    *,
    sweep_dir: Path,
    filename: str,
    record: dict[str, object],
) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    p = sweep_dir / filename
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_progress_txt(
    *,
    sweep_dir: Path,
    completed: int,
    total: int,
    elapsed_s: float,
    eta_s: float,
    last: str,
) -> None:
    p = sweep_dir / "progress.txt"
    rate = completed / elapsed_s if elapsed_s > 0 else 0.0
    eta_at = (datetime.now() + timedelta(seconds=float(eta_s))) if eta_s > 0 else datetime.now()
    p.write_text(
        f"completed={completed}/{total} elapsed_s={elapsed_s:.1f} eta_s={eta_s:.1f} eta_at={eta_at.isoformat(timespec='seconds')} rate_runs_per_s={rate:.3f} last={last}\n",
        encoding="utf-8",
    )


def _sweep_worker(payload: tuple[dict, float, float, int]) -> dict[str, object]:
    base_dict, sr, w, seed = payload
    cfg = ExperimentConfig(
        world=WorldConfig(**base_dict["world"]),
        evaders=EvaderConfig(**base_dict["evaders"]),
        pursuers=PursuerConfig(**base_dict["pursuers"]),
        safe_zones=SafeZoneConfig(**base_dict["safe_zones"]),
        run=RunConfig(**base_dict["run"]),
    )

    cfg = replace(
        cfg,
        pursuers=replace(cfg.pursuers, speed_ratio=float(sr)),
        evaders=replace(cfg.evaders, w_align=float(w)),
        run=replace(cfg.run, seed=int(seed), save_timeseries=False, save_events=False),
    )
    summary = run_summary(cfg)
    return {"speed_ratio": float(sr), "w_align": float(w), "seed": int(seed), **summary}


def _noise_worker(payload: tuple[dict, float, float, int]) -> dict[str, object]:
    """
    (base_config_dict, speed_ratio, angle_noise, seed) -> summary row
    """
    base_dict, sr, noise, seed = payload
    cfg = ExperimentConfig(
        world=WorldConfig(**base_dict["world"]),
        evaders=EvaderConfig(**base_dict["evaders"]),
        pursuers=PursuerConfig(**base_dict["pursuers"]),
        safe_zones=SafeZoneConfig(**base_dict["safe_zones"]),
        run=RunConfig(**base_dict["run"]),
    )
    cfg = replace(
        cfg,
        pursuers=replace(cfg.pursuers, speed_ratio=float(sr)),
        evaders=replace(cfg.evaders, angle_noise=float(noise)),
        run=replace(cfg.run, seed=int(seed), save_timeseries=False, save_events=False),
    )
    summary = run_summary(cfg)
    return {"speed_ratio": float(sr), "angle_noise": float(noise), "seed": int(seed), **summary}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    known_cmds = {
        "run",
        "sweep-w-align",
        "sweep-grid",
        "sweep-noise",
        "report",
        "phase-sweep-noise",
        "report-noise",
        "report-pred-entropy",
        "bench-speed",
    }
    if not argv:
        argv = ["run"]
    elif argv[0].startswith("-") or argv[0] not in known_cmds:
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    cfg = load_config(args.config) if getattr(args, "config", None) else ExperimentConfig()

    if getattr(args, "out", None) is not None:
        cfg = replace(cfg, run=replace(cfg.run, out_dir=str(args.out)))

    if cmd == "run":
        if args.seed is not None:
            cfg = replace(cfg, run=replace(cfg.run, seed=int(args.seed)))
        if args.steps is not None:
            cfg = replace(cfg, run=replace(cfg.run, steps=int(args.steps)))

        summary, run_dir = run_once(cfg)
        print(f"Saved run to: {run_dir}")
        print(f"Summary: {summary}")
        return 0

    if cmd == "sweep-w-align":
        if args.steps is not None:
            cfg = replace(cfg, run=replace(cfg.run, steps=int(args.steps)))
        cfg = replace(
            cfg,
            run=replace(cfg.run, save_timeseries=not args.no_save_timeseries, save_events=not args.no_save_events),
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = ensure_dir(Path(cfg.run.out_dir) / f"sweep_{stamp}_w_align")
        json_dump(sweep_dir / "base_config.json", cfg.to_dict())

        results: list[dict[str, object]] = []
        for w in args.values:
            for seed in args.seeds:
                cfg_run = replace(
                    cfg,
                    evaders=replace(cfg.evaders, w_align=float(w)),
                    run=replace(cfg.run, seed=int(seed), out_dir=str(sweep_dir)),
                )
                run_name = f"w_align{float(w):.3f}_seed{int(seed)}"
                summary, run_dir = run_once(cfg_run, run_name=run_name)
                results.append(
                    {
                        "w_align": float(w),
                        "seed": int(seed),
                        "run_dir": str(run_dir),
                        **summary,
                    }
                )

        csv_write(
            sweep_dir / "results.csv",
            results,
            fieldnames=list(results[0].keys()) if results else ["w_align", "seed"],
        )
        print(f"Saved sweep to: {sweep_dir}")
        print(f"Runs: {len(results)}")
        return 0

    if cmd == "sweep-grid":
        if args.steps is not None:
            cfg = replace(cfg, run=replace(cfg.run, steps=int(args.steps)))
        cfg = replace(
            cfg,
            run=replace(cfg.run, save_timeseries=not args.no_save_timeseries, save_events=not args.no_save_events),
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = ensure_dir(Path(cfg.run.out_dir) / f"sweep_{stamp}_grid")
        json_dump(sweep_dir / "base_config.json", cfg.to_dict())

        total = len(args.speed_ratio) * len(args.w_align) * len(args.seeds)
        start_t = time.monotonic()
        completed = 0

        base_dict = cfg.to_dict()
        _write_progress(
            sweep_dir=sweep_dir,
            filename=args.progress_file,
            record={"ts": datetime.now().isoformat(timespec="seconds"), "type": "start", "total": total, "jobs": int(args.jobs)},
        )

        results: list[dict[str, object]] = []

        def on_done_grid(row: dict[str, object], last: str) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.monotonic() - start_t
            avg = elapsed / completed if completed else 0.0
            eta = avg * (total - completed)
            _write_progress(
                sweep_dir=sweep_dir,
                filename=args.progress_file,
                record={
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "type": "done",
                    "completed": completed,
                    "total": total,
                    "elapsed_s": round(elapsed, 3),
                    "eta_s": round(eta, 3),
                    "last": last,
                },
            )
            _write_progress_txt(sweep_dir=sweep_dir, completed=completed, total=total, elapsed_s=elapsed, eta_s=eta, last=last)

        if int(args.jobs) <= 1:
            for sr in args.speed_ratio:
                for w in args.w_align:
                    for seed in args.seeds:
                        cfg_run = replace(
                            cfg,
                            pursuers=replace(cfg.pursuers, speed_ratio=float(sr)),
                            evaders=replace(cfg.evaders, w_align=float(w)),
                            run=replace(cfg.run, seed=int(seed), out_dir=str(sweep_dir)),
                        )
                        last = f"sr={float(sr):g}, w_align={float(w):g}, seed={int(seed)}"
                        run_name = f"sr{float(sr):.3f}_w{float(w):.3f}_seed{int(seed)}"
                        summary, run_dir = run_once(cfg_run, run_name=run_name, save_run=bool(args.save_runs))
                        row: dict[str, object] = {
                            "speed_ratio": float(sr),
                            "w_align": float(w),
                            "seed": int(seed),
                            "run_dir": str(run_dir) if run_dir is not None else "",
                            **summary,
                        }
                        results.append(row)
                        on_done_grid(row, last)
        else:
            jobs = int(args.jobs)
            payloads: list[tuple[dict, float, float, int]] = []
            for sr in args.speed_ratio:
                for w in args.w_align:
                    for seed in args.seeds:
                        payloads.append((base_dict, float(sr), float(w), int(seed)))

            with ProcessPoolExecutor(max_workers=jobs) as ex:
                fut_to_params = {ex.submit(_sweep_worker, pl): (pl[1], pl[2], pl[3]) for pl in payloads}
                for fut in as_completed(fut_to_params):
                    sr, w, seed = fut_to_params[fut]
                    row = fut.result()
                    row["run_dir"] = ""
                    results.append(row)
                    last = f"sr={float(sr):g}, w_align={float(w):g}, seed={int(seed)}"
                    on_done_grid(row, last)

        csv_write(
            sweep_dir / "results.csv",
            results,
            fieldnames=list(results[0].keys()) if results else ["speed_ratio", "w_align", "seed"],
        )
        print(f"Saved sweep to: {sweep_dir}")
        print(f"Runs: {len(results)}")
        return 0

    if cmd == "report":
        sweep_dir = Path(args.sweep_dir)
        out_dir = ensure_dir(args.out_dir)
        rows = load_results_csv(sweep_dir / "results.csv")
        grouped = summarize_by_group(rows)

        # Flatten grouped stats into rows for plotting + csv
        grouped_rows: list[dict[str, Any]] = []
        for (sr, wa), stats in grouped.items():
            grouped_rows.append(
                {
                    "speed_ratio": sr,
                    "w_align": wa,
                    "n": stats["safe_frac"].n,
                    "safe_mean": stats["safe_frac"].mean,
                    "safe_ci95": stats["safe_frac"].ci95,
                    "captured_mean": stats["captured_frac"].mean,
                    "captured_ci95": stats["captured_frac"].ci95,
                    "P_mean": stats["P_mean"].mean,
                    "P_mean_ci95": stats["P_mean"].ci95,
                    "P_var": stats["P_var"].mean,
                    "P_var_ci95": stats["P_var"].ci95,
                    "chi_mean": stats["chi"].mean,
                    "chi_ci95": stats["chi"].ci95,
                    "rho1_P_mean": stats["rho1_P"].mean,
                    "rho1_P_ci95": stats["rho1_P"].ci95,
                    "tau_P_ar1_mean": stats["tau_P_ar1"].mean,
                    "tau_P_ar1_ci95": stats["tau_P_ar1"].ci95,
                    "xi_fluct_mean": stats["xi_fluct"].mean,
                    "xi_fluct_ci95": stats["xi_fluct"].ci95,
                    "components_mean": stats["components"].mean,
                    "components_ci95": stats["components"].ci95,
                    "chi_local_mean": stats["chi_local"].mean,
                    "chi_local_ci95": stats["chi_local"].ci95,
                }
            )

        write_group_summary_csv(out_dir / "group_summary.csv", grouped)

        figs_dir = ensure_dir(Path(out_dir) / "figs")
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="safe_mean",
            metric_ci="safe_ci95",
            out_path=figs_dir / "safe_vs_w_align.png",
            title="Safe fraction vs w_align (mean ± 95% CI)",
            ylabel="safe_frac",
        )
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="captured_mean",
            metric_ci="captured_ci95",
            out_path=figs_dir / "captured_vs_w_align.png",
            title="Captured fraction vs w_align (mean ± 95% CI)",
            ylabel="captured_frac",
        )
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="P_mean",
            metric_ci="P_mean_ci95",
            out_path=figs_dir / "P_mean_vs_w_align.png",
            title="Polarization mean vs w_align (mean ± 95% CI)",
            ylabel="P_mean",
        )
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="chi_mean",
            metric_ci="chi_ci95",
            out_path=figs_dir / "chi_vs_w_align.png",
            title="Susceptibility proxy χ=N·Var(P) vs w_align (mean ± 95% CI)",
            ylabel="chi",
        )
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="chi_local_mean",
            metric_ci="chi_local_ci95",
            out_path=figs_dir / "chi_local_vs_w_align.png",
            title="Local susceptibility proxy χ_local=N·Var(P_local) vs w_align (mean ± 95% CI)",
            ylabel="chi_local",
        )
        plot_metric_vs_w_align(
            grouped_rows=grouped_rows,
            metric_mean="tau_P_ar1_mean",
            metric_ci="tau_P_ar1_ci95",
            out_path=figs_dir / "tau_P_ar1_vs_w_align.png",
            title="Correlation time proxy τ(P) (AR(1)) vs w_align (mean ± 95% CI)",
            ylabel="tau_P_ar1",
        )
        plot_heatmap(
            grouped_rows=grouped_rows,
            value_key="safe_mean",
            out_path=figs_dir / "heatmap_safe_mean.png",
            title="Heatmap: safe_frac mean",
        )
        plot_heatmap(
            grouped_rows=grouped_rows,
            value_key="chi_mean",
            out_path=figs_dir / "heatmap_chi_mean.png",
            title="Heatmap: χ mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="chi_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_chi.png",
            title="safe_frac vs χ (group means)",
            xlabel="chi_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="chi_local_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_chi_local.png",
            title="safe_frac vs χ_local (group means)",
            xlabel="chi_local_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="xi_fluct_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_xi.png",
            title="safe_frac vs ξ_fluct (group means)",
            xlabel="xi_fluct_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="tau_P_ar1_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_tau_P_ar1.png",
            title="safe_frac vs τ(P) (group means)",
            xlabel="tau_P_ar1_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="components_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_components.png",
            title="safe_frac vs components (group means)",
            xlabel="components_mean",
            ylabel="safe_mean",
        )

        # Markdown report (reader-facing; no dialog)
        md = []
        md.append(f"# {args.title}\n")
        md.append("## Experiment setup\n")
        md.append("- Pursuer count is fixed by the base config.\n")
        md.append("- Aggregation: mean ± 95% CI across seeds for each (v_p/v_e, w_align).\n")
        def _p(path: Path) -> str:
            return path.as_posix()

        md.append("## Artifacts\n")
        md.append(f"- Sweep directory: `{_p(sweep_dir)}`\n")
        base_cfg = sweep_dir / "base_config.json"
        if base_cfg.exists():
            md.append(f"- Base config: `{_p(base_cfg)}`\n")
        md.append(f"- Group summary (aggregated): `{_p(Path(out_dir) / 'group_summary.csv')}`\n")
        md.append(f"- Figures: `{_p(figs_dir)}`\n")

        md.append("\n## Aggregated summary\n")
        speed_ratios = sorted({float(r['speed_ratio']) for r in grouped_rows})
        md.append("| v_p/v_e | best w (safe) | safe | best w (χ) | χ | best w (χ_local) | χ_local | best w (τ) | τ | best w (ξ) | ξ |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for sr in speed_ratios:
            rs = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
            rs_safe = sorted(rs, key=lambda r: float(r["safe_mean"]), reverse=True)
            rs_chi = sorted(rs, key=lambda r: float(r["chi_mean"]), reverse=True)
            rs_xi = sorted(rs, key=lambda r: float(r["xi_fluct_mean"]), reverse=True)
            rs_chi_local = sorted(rs, key=lambda r: float(r["chi_local_mean"]), reverse=True)
            rs_tau = sorted(rs, key=lambda r: float(r["tau_P_ar1_mean"]), reverse=True)
            if rs_safe and rs_chi and rs_xi and rs_chi_local and rs_tau:
                md.append(
                    f"| {sr:g} | {float(rs_safe[0]['w_align']):g} | {float(rs_safe[0]['safe_mean']):.4f} | {float(rs_chi[0]['w_align']):g} | {float(rs_chi[0]['chi_mean']):.4f} | {float(rs_chi_local[0]['w_align']):g} | {float(rs_chi_local[0]['chi_local_mean']):.4f} | {float(rs_tau[0]['w_align']):g} | {float(rs_tau[0]['tau_P_ar1_mean']):.4f} | {float(rs_xi[0]['w_align']):g} | {float(rs_xi[0]['xi_fluct_mean']):.4f} |\n"
                )

        # Relationship between χ and task success (group means)
        md.append("\n## Criticality–performance relationships (group means)\n")
        md.append("| v_p/v_e | corr(safe, χ) | |w_safe-w_χ| | corr(safe, χ_local) | |w_safe-w_χ_local| | corr(safe, τ) | |w_safe-w_τ| | corr(safe, ξ) | |w_safe-w_ξ| |\n")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for sr in speed_ratios:
            rs = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
            rs_safe = sorted(rs, key=lambda r: float(r["safe_mean"]), reverse=True)
            rs_chi = sorted(rs, key=lambda r: float(r["chi_mean"]), reverse=True)
            rs_xi = sorted(rs, key=lambda r: float(r["xi_fluct_mean"]), reverse=True)
            rs_chi_local = sorted(rs, key=lambda r: float(r["chi_local_mean"]), reverse=True)
            rs_tau = sorted(rs, key=lambda r: float(r["tau_P_ar1_mean"]), reverse=True)
            if not rs_safe or not rs_chi or not rs_xi or not rs_chi_local or not rs_tau:
                continue

            def corr(a: list[float], b: list[float]) -> float:
                ax = sum(a) / len(a)
                by = sum(b) / len(b)
                num = sum((x - ax) * (y - by) for x, y in zip(a, b))
                denx = sum((x - ax) ** 2 for x in a) ** 0.5
                deny = sum((y - by) ** 2 for y in b) ** 0.5
                return num / (denx * deny) if denx > 0 and deny > 0 else 0.0

            ys = [float(r["safe_mean"]) for r in rs]
            corr_chi = corr([float(r["chi_mean"]) for r in rs], ys)
            corr_chi_local = corr([float(r["chi_local_mean"]) for r in rs], ys)
            corr_tau = corr([float(r["tau_P_ar1_mean"]) for r in rs], ys)
            corr_xi = corr([float(r["xi_fluct_mean"]) for r in rs], ys)

            dw = abs(float(rs_safe[0]["w_align"]) - float(rs_chi[0]["w_align"]))
            dw_chi_local = abs(float(rs_safe[0]["w_align"]) - float(rs_chi_local[0]["w_align"]))
            dw_tau = abs(float(rs_safe[0]["w_align"]) - float(rs_tau[0]["w_align"]))
            dw_xi = abs(float(rs_safe[0]["w_align"]) - float(rs_xi[0]["w_align"]))
            md.append(f"| {sr:g} | {corr_chi:.3f} | {dw:.3f} | {corr_chi_local:.3f} | {dw_chi_local:.3f} | {corr_tau:.3f} | {dw_tau:.3f} | {corr_xi:.3f} | {dw_xi:.3f} |\n")
        md.append("\n## Plots\n")
        md.append(f"![safe_vs_w_align](figs/safe_vs_w_align.png)\n")
        md.append(f"![captured_vs_w_align](figs/captured_vs_w_align.png)\n")
        md.append(f"![P_mean_vs_w_align](figs/P_mean_vs_w_align.png)\n")
        md.append(f"![chi_vs_w_align](figs/chi_vs_w_align.png)\n")
        md.append(f"![chi_local_vs_w_align](figs/chi_local_vs_w_align.png)\n")
        md.append(f"![tau_P_ar1_vs_w_align](figs/tau_P_ar1_vs_w_align.png)\n")
        md.append(f"![heatmap_safe_mean](figs/heatmap_safe_mean.png)\n")
        md.append(f"![heatmap_chi_mean](figs/heatmap_chi_mean.png)\n")
        md.append(f"![scatter_safe_vs_chi](figs/scatter_safe_vs_chi.png)\n")
        md.append(f"![scatter_safe_vs_chi_local](figs/scatter_safe_vs_chi_local.png)\n")
        md.append(f"![scatter_safe_vs_xi](figs/scatter_safe_vs_xi.png)\n")
        md.append(f"![scatter_safe_vs_tau_P_ar1](figs/scatter_safe_vs_tau_P_ar1.png)\n")
        md.append(f"![scatter_safe_vs_components](figs/scatter_safe_vs_components.png)\n")

        (Path(out_dir) / "report.md").write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote report to: {out_dir}")
        return 0

    if cmd == "report-pred-entropy":
        report_path = run_predictive_entropy_report(
            sweep_dir=Path(args.sweep_dir),
            out_dir=Path(args.out_dir),
            title=str(args.title),
            bins=int(args.bins),
            ema_span=int(args.ema_span),
            order=int(args.order),
        )
        print(f"Wrote predictive-entropy report to: {report_path}")
        return 0

    if cmd == "phase-sweep-noise":
        if args.steps is not None:
            cfg = replace(cfg, run=replace(cfg.run, steps=int(args.steps)))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = ensure_dir(Path(cfg.run.out_dir) / f"sweep_{stamp}_phase_noise")

        # Phase setting overrides: no pursuers, no safe zones; isolate alignment+noise (+sep)
        phase_cfg = replace(
            cfg,
            pursuers=replace(cfg.pursuers, count=0),
            safe_zones=replace(cfg.safe_zones, active_max=0),
            evaders=replace(cfg.evaders, w_goal=0.0, w_avoid=0.0, w_explore=0.0, w_align=1.0),
            run=replace(cfg.run, out_dir=str(sweep_dir), save_timeseries=False, save_events=False),
        )
        json_dump(sweep_dir / "base_config.json", phase_cfg.to_dict())

        total = len(args.noise) * len(args.seeds)
        start_t = time.monotonic()
        completed = 0
        _write_progress(
            sweep_dir=sweep_dir,
            filename=args.progress_file,
            record={"ts": datetime.now().isoformat(timespec="seconds"), "type": "start", "total": total, "jobs": int(args.jobs)},
        )

        results: list[dict[str, object]] = []
        base_dict = phase_cfg.to_dict()

        def on_done_phase(last: str) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.monotonic() - start_t
            avg = elapsed / completed if completed else 0.0
            eta = avg * (total - completed)
            _write_progress(
                sweep_dir=sweep_dir,
                filename=args.progress_file,
                record={
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "type": "done",
                    "completed": completed,
                    "total": total,
                    "elapsed_s": round(elapsed, 3),
                    "eta_s": round(eta, 3),
                    "last": last,
                },
            )
            _write_progress_txt(sweep_dir=sweep_dir, completed=completed, total=total, elapsed_s=elapsed, eta_s=eta, last=last)

        jobs = int(args.jobs)
        if jobs <= 1:
            for noise in args.noise:
                for seed in args.seeds:
                    cfg_run = replace(
                        phase_cfg,
                        evaders=replace(phase_cfg.evaders, angle_noise=float(noise)),
                        run=replace(phase_cfg.run, seed=int(seed)),
                    )
                    summary = run_summary(cfg_run)
                    results.append({"speed_ratio": 0.0, "angle_noise": float(noise), "seed": int(seed), **summary})
                    on_done_phase(f"noise={float(noise):g}, seed={int(seed)}")
        else:
            payloads: list[tuple[dict, float, float, int]] = []
            for noise in args.noise:
                for seed in args.seeds:
                    payloads.append((base_dict, 0.0, float(noise), int(seed)))

            with ProcessPoolExecutor(max_workers=jobs) as ex:
                fut_to_params = {ex.submit(_noise_worker, pl): (pl[2], pl[3]) for pl in payloads}
                for fut in as_completed(fut_to_params):
                    noise, seed = fut_to_params[fut]
                    row = fut.result()
                    results.append(row)
                    on_done_phase(f"noise={float(noise):g}, seed={int(seed)}")

        csv_write(
            sweep_dir / "results.csv",
            results,
            fieldnames=list(results[0].keys()) if results else ["angle_noise", "seed"],
        )
        print(f"Saved sweep to: {sweep_dir}")
        print(f"Runs: {len(results)}")
        return 0

    if cmd == "report-noise":
        sweep_dir = Path(args.sweep_dir)
        out_dir = ensure_dir(args.out_dir)
        rows = load_results_csv(sweep_dir / "results.csv")

        grouped = summarize_by_two_keys(rows, key_a="speed_ratio", key_b="angle_noise")
        grouped_rows: list[dict[str, Any]] = []
        for (sr, noise), stats in grouped.items():
            grouped_rows.append(
                {
                    "speed_ratio": sr,
                    "angle_noise": noise,
                    "n": stats["safe_frac"].n,
                    "safe_mean": stats["safe_frac"].mean,
                    "safe_ci95": stats["safe_frac"].ci95,
                    "captured_mean": stats["captured_frac"].mean,
                    "captured_ci95": stats["captured_frac"].ci95,
                    "P_mean": stats["P_mean"].mean,
                    "P_mean_ci95": stats["P_mean"].ci95,
                    "chi_mean": stats["chi"].mean,
                    "chi_ci95": stats["chi"].ci95,
                    "rho1_P_mean": stats["rho1_P"].mean,
                    "rho1_P_ci95": stats["rho1_P"].ci95,
                    "tau_P_ar1_mean": stats["tau_P_ar1"].mean,
                    "tau_P_ar1_ci95": stats["tau_P_ar1"].ci95,
                    "xi_fluct_mean": stats["xi_fluct"].mean,
                    "xi_fluct_ci95": stats["xi_fluct"].ci95,
                    "components_mean": stats["components"].mean,
                    "components_ci95": stats["components"].ci95,
                    "chi_local_mean": stats["chi_local"].mean,
                    "chi_local_ci95": stats["chi_local"].ci95,
                }
            )

        grouped_rows.sort(key=lambda r: (float(r["speed_ratio"]), float(r["angle_noise"])))
        csv_write(
            Path(out_dir) / "group_summary.csv",
            grouped_rows,
            fieldnames=[
                "speed_ratio",
                "angle_noise",
                "n",
                "safe_mean",
                "safe_ci95",
                "captured_mean",
                "captured_ci95",
                "P_mean",
                "P_mean_ci95",
                "chi_mean",
                "chi_ci95",
                "chi_local_mean",
                "chi_local_ci95",
                "rho1_P_mean",
                "rho1_P_ci95",
                "tau_P_ar1_mean",
                "tau_P_ar1_ci95",
                "xi_fluct_mean",
                "xi_fluct_ci95",
                "components_mean",
                "components_ci95",
            ],
        )

        figs_dir = ensure_dir(Path(out_dir) / "figs")
        plot_metric_vs_x(
            grouped_rows=grouped_rows,
            x_key="angle_noise",
            metric_mean="safe_mean",
            metric_ci="safe_ci95",
            out_path=figs_dir / "safe_vs_noise.png",
            title="safe_frac vs angle_noise (mean ± 95% CI)",
            xlabel="angle_noise (rad)",
            ylabel="safe_frac",
        )
        plot_metric_vs_x(
            grouped_rows=grouped_rows,
            x_key="angle_noise",
            metric_mean="chi_mean",
            metric_ci="chi_ci95",
            out_path=figs_dir / "chi_vs_noise.png",
            title="χ=N·Var(P) vs angle_noise (mean ± 95% CI)",
            xlabel="angle_noise (rad)",
            ylabel="chi",
        )
        plot_metric_vs_x(
            grouped_rows=grouped_rows,
            x_key="angle_noise",
            metric_mean="chi_local_mean",
            metric_ci="chi_local_ci95",
            out_path=figs_dir / "chi_local_vs_noise.png",
            title="χ_local=N·Var(P_local) vs angle_noise (mean ± 95% CI)",
            xlabel="angle_noise (rad)",
            ylabel="chi_local",
        )
        plot_metric_vs_x(
            grouped_rows=grouped_rows,
            x_key="angle_noise",
            metric_mean="P_mean",
            metric_ci="P_mean_ci95",
            out_path=figs_dir / "P_mean_vs_noise.png",
            title="P_mean vs angle_noise (mean ± 95% CI)",
            xlabel="angle_noise (rad)",
            ylabel="P_mean",
        )
        plot_metric_vs_x(
            grouped_rows=grouped_rows,
            x_key="angle_noise",
            metric_mean="tau_P_ar1_mean",
            metric_ci="tau_P_ar1_ci95",
            out_path=figs_dir / "tau_P_ar1_vs_noise.png",
            title="τ(P) (AR(1)) vs angle_noise (mean ± 95% CI)",
            xlabel="angle_noise (rad)",
            ylabel="tau_P_ar1",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="chi_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_chi.png",
            title="safe_frac vs χ (group means)",
            xlabel="chi_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="chi_local_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_chi_local.png",
            title="safe_frac vs χ_local (group means)",
            xlabel="chi_local_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="xi_fluct_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_xi.png",
            title="safe_frac vs ξ_fluct (group means)",
            xlabel="xi_fluct_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="tau_P_ar1_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_tau_P_ar1.png",
            title="safe_frac vs τ(P) (group means)",
            xlabel="tau_P_ar1_mean",
            ylabel="safe_mean",
        )
        plot_scatter(
            grouped_rows=grouped_rows,
            x_key="components_mean",
            y_key="safe_mean",
            out_path=figs_dir / "scatter_safe_vs_components.png",
            title="safe_frac vs components (group means)",
            xlabel="components_mean",
            ylabel="safe_mean",
        )

        md = []
        md.append(f"# {args.title}\n")
        md.append("## Artifacts\n")
        md.append(f"- Sweep directory: `{sweep_dir.as_posix()}`\n")
        base_cfg = sweep_dir / "base_config.json"
        if base_cfg.exists():
            md.append(f"- Base config: `{base_cfg.as_posix()}`\n")
        md.append(f"- Figures: `{figs_dir.as_posix()}`\n")
        md.append("\n## Plots\n")
        md.append("![safe_vs_noise](figs/safe_vs_noise.png)\n")
        md.append("![chi_vs_noise](figs/chi_vs_noise.png)\n")
        md.append("![chi_local_vs_noise](figs/chi_local_vs_noise.png)\n")
        md.append("![P_mean_vs_noise](figs/P_mean_vs_noise.png)\n")
        md.append("![tau_P_ar1_vs_noise](figs/tau_P_ar1_vs_noise.png)\n")
        md.append("![scatter_safe_vs_chi](figs/scatter_safe_vs_chi.png)\n")
        md.append("![scatter_safe_vs_chi_local](figs/scatter_safe_vs_chi_local.png)\n")
        md.append("![scatter_safe_vs_xi](figs/scatter_safe_vs_xi.png)\n")
        md.append("![scatter_safe_vs_tau_P_ar1](figs/scatter_safe_vs_tau_P_ar1.png)\n")
        md.append("![scatter_safe_vs_components](figs/scatter_safe_vs_components.png)\n")
        (Path(out_dir) / "report.md").write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote report to: {out_dir}")
        return 0

    if cmd == "sweep-noise":
        if args.steps is not None:
            cfg = replace(cfg, run=replace(cfg.run, steps=int(args.steps)))
        if args.w_align is not None:
            cfg = replace(cfg, evaders=replace(cfg.evaders, w_align=float(args.w_align)))

        cfg = replace(
            cfg,
            run=replace(cfg.run, save_timeseries=not args.no_save_timeseries, save_events=not args.no_save_events),
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = ensure_dir(Path(cfg.run.out_dir) / f"sweep_{stamp}_noise")
        cfg = replace(cfg, run=replace(cfg.run, out_dir=str(sweep_dir)))
        json_dump(sweep_dir / "base_config.json", cfg.to_dict())

        total = len(args.speed_ratio) * len(args.noise) * len(args.seeds)
        start_t = time.monotonic()
        completed = 0
        _write_progress(
            sweep_dir=sweep_dir,
            filename=args.progress_file,
            record={"ts": datetime.now().isoformat(timespec="seconds"), "type": "start", "total": total, "jobs": int(args.jobs)},
        )

        results: list[dict[str, object]] = []

        def on_done_noise(last: str) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.monotonic() - start_t
            avg = elapsed / completed if completed else 0.0
            eta = avg * (total - completed)
            _write_progress(
                sweep_dir=sweep_dir,
                filename=args.progress_file,
                record={
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "type": "done",
                    "completed": completed,
                    "total": total,
                    "elapsed_s": round(elapsed, 3),
                    "eta_s": round(eta, 3),
                    "last": last,
                },
            )
            _write_progress_txt(sweep_dir=sweep_dir, completed=completed, total=total, elapsed_s=elapsed, eta_s=eta, last=last)

        jobs = int(args.jobs)
        if jobs <= 1:
            for sr in args.speed_ratio:
                for noise in args.noise:
                    for seed in args.seeds:
                        cfg_run = replace(
                            cfg,
                            pursuers=replace(cfg.pursuers, speed_ratio=float(sr)),
                            evaders=replace(cfg.evaders, angle_noise=float(noise)),
                            run=replace(cfg.run, seed=int(seed)),
                        )
                        last = f"sr={float(sr):g}, noise={float(noise):g}, seed={int(seed)}"
                        run_name = f"sr{float(sr):.3f}_noise{float(noise):.3f}_seed{int(seed)}"
                        summary, run_dir = run_once(cfg_run, run_name=run_name, save_run=bool(args.save_runs))
                        row: dict[str, object] = {
                            "speed_ratio": float(sr),
                            "angle_noise": float(noise),
                            "seed": int(seed),
                            "run_dir": str(run_dir) if run_dir is not None else "",
                            **summary,
                        }
                        results.append(row)
                        on_done_noise(last)
        else:
            base_dict = cfg.to_dict()
            payloads: list[tuple[dict, float, float, int]] = []
            for sr in args.speed_ratio:
                for noise in args.noise:
                    for seed in args.seeds:
                        payloads.append((base_dict, float(sr), float(noise), int(seed)))

            with ProcessPoolExecutor(max_workers=jobs) as ex:
                fut_to_params = {ex.submit(_noise_worker, pl): (pl[1], pl[2], pl[3]) for pl in payloads}
                for fut in as_completed(fut_to_params):
                    sr, noise, seed = fut_to_params[fut]
                    row = fut.result()
                    row["run_dir"] = ""
                    results.append(row)
                    on_done_noise(f"sr={float(sr):g}, noise={float(noise):g}, seed={int(seed)}")

        csv_write(
            sweep_dir / "results.csv",
            results,
            fieldnames=list(results[0].keys()) if results else ["speed_ratio", "angle_noise", "seed"],
        )
        print(f"Saved sweep to: {sweep_dir}")
        print(f"Runs: {len(results)}")
        return 0

    if cmd == "bench-speed":
        result = run_small_benchmark(
            cfg,
            iterations=int(args.iterations),
            steps=int(args.steps),
            repeats=int(args.repeats),
            seed=int(args.seed),
        )
        print("Benchmark summary:")
        print(f"- numba_available: {result['numba_available']}")
        print(f"- evader_step numpy mean (s): {result['evader_step_numpy_mean_s']:.6f}")
        print(f"- evader_step numba mean (s): {result['evader_step_numba_mean_s']:.6f}")
        print(f"- evader_step speedup (x): {result['evader_step_speedup_x']:.3f}")
        print(f"- simulation numpy mean (s): {result['simulation_numpy_mean_s']:.6f}")
        print(f"- simulation numba mean (s): {result['simulation_numba_mean_s']:.6f}")
        print(f"- simulation speedup (x): {result['simulation_speedup_x']:.3f}")
        behavior = result["behavior_check"]
        print(f"- behavior count mismatches: {int(behavior['count_mismatch_runs'])}")
        print(f"- behavior max safe_frac delta: {float(behavior['max_safe_frac_delta']):.6e}")
        print(f"- behavior max chi delta: {float(behavior['max_chi_delta']):.6e}")

        out_path: Path
        if args.json_out is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(cfg.run.out_dir) / f"bench_{stamp}.json"
        else:
            out_path = Path(args.json_out)
        json_dump(out_path, result)
        print(f"Saved benchmark json: {out_path}")
        return 0

    raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
