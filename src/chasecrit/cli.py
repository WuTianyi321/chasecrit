from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

from .config import ExperimentConfig, load_config
from .io_utils import csv_write, ensure_dir, json_dump
from .report import load_results_csv, summarize_by_group, write_group_summary_csv
from .plotting import plot_heatmap, plot_metric_vs_w_align, plot_scatter
from .sim import run_once


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

    rep = sub.add_parser("report", help="Create plots and a markdown report from a sweep directory")
    rep.add_argument("--sweep-dir", type=str, required=True, help="Sweep directory containing results.csv")
    rep.add_argument("--out-dir", type=str, required=True, help="Output directory for report and figures")
    rep.add_argument("--title", type=str, default="Fixed pursuer-count scan", help="Report title")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    known_cmds = {"run", "sweep-w-align", "sweep-grid", "report"}
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

        results: list[dict[str, object]] = []
        for sr in args.speed_ratio:
            for w in args.w_align:
                for seed in args.seeds:
                    cfg_run = replace(
                        cfg,
                        pursuers=replace(cfg.pursuers, speed_ratio=float(sr)),
                        evaders=replace(cfg.evaders, w_align=float(w)),
                        run=replace(cfg.run, seed=int(seed), out_dir=str(sweep_dir)),
                    )
                    run_name = f"sr{float(sr):.3f}_w{float(w):.3f}_seed{int(seed)}"
                    summary, run_dir = run_once(cfg_run, run_name=run_name)
                    results.append(
                        {
                            "speed_ratio": float(sr),
                            "w_align": float(w),
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            **summary,
                        }
                    )

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
        md.append("| v_p/v_e | best w_align (safe_frac) | safe_frac mean | best w_align (χ) | χ mean |\n")
        md.append("|---:|---:|---:|---:|---:|\n")
        for sr in speed_ratios:
            rs = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
            rs_safe = sorted(rs, key=lambda r: float(r["safe_mean"]), reverse=True)
            rs_chi = sorted(rs, key=lambda r: float(r["chi_mean"]), reverse=True)
            if rs_safe and rs_chi:
                md.append(
                    f"| {sr:g} | {float(rs_safe[0]['w_align']):g} | {float(rs_safe[0]['safe_mean']):.4f} | {float(rs_chi[0]['w_align']):g} | {float(rs_chi[0]['chi_mean']):.4f} |\n"
                )

        # Relationship between χ and task success (group means)
        md.append("\n## χ–performance relationship (group means)\n")
        md.append("| v_p/v_e | corr(safe_mean, χ_mean) | |w_safe - w_χ| |\n")
        md.append("|---:|---:|---:|\n")
        for sr in speed_ratios:
            rs = [r for r in grouped_rows if float(r["speed_ratio"]) == sr]
            rs_safe = sorted(rs, key=lambda r: float(r["safe_mean"]), reverse=True)
            rs_chi = sorted(rs, key=lambda r: float(r["chi_mean"]), reverse=True)
            if not rs_safe or not rs_chi:
                continue

            xs = [float(r["chi_mean"]) for r in rs]
            ys = [float(r["safe_mean"]) for r in rs]
            # Pearson correlation (population)
            x = sum(xs) / len(xs)
            y = sum(ys) / len(ys)
            num = sum((a - x) * (b - y) for a, b in zip(xs, ys))
            denx = sum((a - x) ** 2 for a in xs) ** 0.5
            deny = sum((b - y) ** 2 for b in ys) ** 0.5
            corr = num / (denx * deny) if denx > 0 and deny > 0 else 0.0

            dw = abs(float(rs_safe[0]["w_align"]) - float(rs_chi[0]["w_align"]))
            md.append(f"| {sr:g} | {corr:.3f} | {dw:.3f} |\n")
        md.append("\n## Plots\n")
        md.append(f"![safe_vs_w_align](figs/safe_vs_w_align.png)\n")
        md.append(f"![captured_vs_w_align](figs/captured_vs_w_align.png)\n")
        md.append(f"![P_mean_vs_w_align](figs/P_mean_vs_w_align.png)\n")
        md.append(f"![chi_vs_w_align](figs/chi_vs_w_align.png)\n")
        md.append(f"![heatmap_safe_mean](figs/heatmap_safe_mean.png)\n")
        md.append(f"![heatmap_chi_mean](figs/heatmap_chi_mean.png)\n")
        md.append(f"![scatter_safe_vs_chi](figs/scatter_safe_vs_chi.png)\n")

        (Path(out_dir) / "report.md").write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote report to: {out_dir}")
        return 0

    raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
