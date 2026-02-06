from __future__ import annotations

import pytest

from chasecrit.config import ExperimentConfig
from dataclasses import replace
from chasecrit.sim import run_summary


def test_phase_setting_regression_summary() -> None:
    """
    Regression guard for the pure swarm dynamics (no pursuers, no safe zones).
    This keeps the behavioral baseline stable across performance refactors.
    """
    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        evaders=replace(cfg.evaders, w_goal=0.0, w_avoid=0.0, w_explore=0.0, w_align=1.0, angle_noise=1.8),
        pursuers=replace(cfg.pursuers, count=0),
        safe_zones=replace(cfg.safe_zones, active_max=0),
        run=replace(cfg.run, steps=300, seed=0, save_timeseries=False, save_events=False),
    )

    s = run_summary(cfg, prefer_numba=False)
    assert s["safe"] == 0
    assert s["captured"] == 0
    assert s["steps_recorded"] == 300
    assert s["P_mean"] == pytest.approx(0.28214166929301526, abs=1e-9)
    assert s["chi"] == pytest.approx(2.8502630399262796, abs=1e-9)
    assert s["xi_fluct"] == pytest.approx(7.5, abs=1e-12)
    assert s["components"] == 106
    assert s["chi_local"] == pytest.approx(24.57215161676656, abs=1e-9)


def test_task_setting_regression_summary() -> None:
    """
    Regression guard for the full task with safe zones + pursuers.
    """
    cfg = ExperimentConfig()
    cfg = replace(cfg, run=replace(cfg.run, steps=200, seed=0, save_timeseries=False, save_events=False))

    s = run_summary(cfg, prefer_numba=False)
    assert s["steps_recorded"] == 200
    assert s["safe"] == 33
    assert s["captured"] == 6
    assert s["safe_frac"] == pytest.approx(0.2578125, abs=1e-12)
    assert s["chi"] == pytest.approx(1.0009266572971205, abs=1e-9)
    assert s["chi_local"] == pytest.approx(0.207084152082043, abs=1e-9)
