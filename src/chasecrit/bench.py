from __future__ import annotations

from dataclasses import replace
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .geometry import apply_periodic, apply_reflecting, unit
from .policies import EvaderDenseWorkspace, evader_step, numba_runtime_available
from .sim import run_summary


def _bench_evader_step_once(
    cfg: ExperimentConfig,
    *,
    iterations: int,
    seed: int,
    prefer_numba: bool,
) -> float:
    rng_init = np.random.default_rng(seed)
    world_size = np.array([cfg.world.width, cfg.world.height], dtype=np.float64)
    periodic = cfg.world.boundary == "periodic"
    dt = float(cfg.run.dt)

    ne = int(cfg.evaders.count)
    npurs = int(cfg.pursuers.count)
    nz = max(1, int(cfg.safe_zones.active_max))

    pos_e = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(ne, 2)).astype(np.float64)
    vel_e = cfg.evaders.speed * unit(rng_init.normal(size=(ne, 2))).astype(np.float64)
    alive_mask = np.ones((ne,), dtype=bool)

    pos_p = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(npurs, 2)).astype(np.float64)
    pos_z = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(nz, 2)).astype(np.float64)
    zone_active = np.ones((nz,), dtype=bool)

    rng = np.random.default_rng(seed + 1000)
    ws = EvaderDenseWorkspace.create(ne)

    start = perf_counter()
    for _ in range(iterations):
        vel_e = evader_step(
            pos_e=pos_e,
            vel_e=vel_e,
            alive_mask=alive_mask,
            pos_p=pos_p,
            pos_z=pos_z,
            zone_active=zone_active,
            world_size=world_size,
            periodic=periodic,
            rng=rng,
            v_e=cfg.evaders.speed,
            inertia=cfg.evaders.inertia,
            r_nbr=cfg.evaders.neighbor_radius,
            r_pred=cfg.evaders.pred_radius,
            r_detect=cfg.evaders.detect_radius,
            r_sep=cfg.evaders.sep_radius,
            w_align=cfg.evaders.w_align,
            w_goal=cfg.evaders.w_goal,
            w_avoid=cfg.evaders.w_avoid,
            w_explore=cfg.evaders.w_explore,
            sep_strength=cfg.evaders.sep_strength,
            angle_noise=cfg.evaders.angle_noise,
            dense_ws=ws,
            prefer_numba=prefer_numba,
        )
        pos_e = pos_e + vel_e * dt
        if periodic:
            pos_e = apply_periodic(pos_e, world_size)
        else:
            pos_e, vel_e = apply_reflecting(pos_e, vel_e, world_size)

    return perf_counter() - start


def run_small_benchmark(
    cfg: ExperimentConfig,
    *,
    iterations: int = 300,
    steps: int = 300,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "numba_available": bool(numba_runtime_available()),
        "iterations": int(iterations),
        "steps": int(steps),
        "repeats": int(repeats),
        "seed": int(seed),
    }

    # Warm up JIT to avoid counting compilation in benchmark.
    if numba_runtime_available():
        _bench_evader_step_once(cfg, iterations=5, seed=seed, prefer_numba=True)
        warm_cfg = replace(cfg, run=replace(cfg.run, steps=16, seed=seed, save_timeseries=False, save_events=False))
        run_summary(warm_cfg, prefer_numba=True)

    ev_numpy: list[float] = []
    ev_numba: list[float] = []

    sim_numpy: list[float] = []
    sim_numba: list[float] = []
    behavior_rows: list[dict[str, Any]] = []

    for r in range(repeats):
        s = seed + r
        ev_numpy.append(_bench_evader_step_once(cfg, iterations=iterations, seed=s, prefer_numba=False))
        ev_numba.append(_bench_evader_step_once(cfg, iterations=iterations, seed=s, prefer_numba=True))

        cfg_run = replace(cfg, run=replace(cfg.run, steps=steps, seed=s, save_timeseries=False, save_events=False))

        t0 = perf_counter()
        summary_np = run_summary(cfg_run, prefer_numba=False)
        sim_numpy.append(perf_counter() - t0)

        t1 = perf_counter()
        summary_nb = run_summary(cfg_run, prefer_numba=True)
        sim_numba.append(perf_counter() - t1)

        behavior_rows.append(
            {
                "seed": s,
                "safe_np": int(summary_np["safe"]),
                "safe_nb": int(summary_nb["safe"]),
                "captured_np": int(summary_np["captured"]),
                "captured_nb": int(summary_nb["captured"]),
                "safe_frac_np": float(summary_np["safe_frac"]),
                "safe_frac_nb": float(summary_nb["safe_frac"]),
                "chi_np": float(summary_np["chi"]),
                "chi_nb": float(summary_nb["chi"]),
            }
        )

    ev_np_mean = mean(ev_numpy)
    ev_nb_mean = mean(ev_numba)
    sim_np_mean = mean(sim_numpy)
    sim_nb_mean = mean(sim_numba)

    out["evader_step_numpy_mean_s"] = ev_np_mean
    out["evader_step_numba_mean_s"] = ev_nb_mean
    out["evader_step_speedup_x"] = (ev_np_mean / ev_nb_mean) if ev_nb_mean > 0 else 0.0

    out["simulation_numpy_mean_s"] = sim_np_mean
    out["simulation_numba_mean_s"] = sim_nb_mean
    out["simulation_speedup_x"] = (sim_np_mean / sim_nb_mean) if sim_nb_mean > 0 else 0.0

    max_safe_frac_delta = max(abs(x["safe_frac_np"] - x["safe_frac_nb"]) for x in behavior_rows)
    max_chi_delta = max(abs(x["chi_np"] - x["chi_nb"]) for x in behavior_rows)
    mismatched_counts = sum(
        1
        for x in behavior_rows
        if x["safe_np"] != x["safe_nb"] or x["captured_np"] != x["captured_nb"]
    )

    out["behavior_check"] = {
        "max_safe_frac_delta": max_safe_frac_delta,
        "max_chi_delta": max_chi_delta,
        "count_mismatch_runs": int(mismatched_counts),
        "rows": behavior_rows,
    }
    return out
