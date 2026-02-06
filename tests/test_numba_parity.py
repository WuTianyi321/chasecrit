from __future__ import annotations

from dataclasses import replace

import pytest

from chasecrit.config import ExperimentConfig
from chasecrit.policies import numba_runtime_available
from chasecrit.sim import run_summary


@pytest.mark.skipif(not numba_runtime_available(), reason="numba not available")
def test_numba_path_preserves_task_outcomes_on_small_suite() -> None:
    base = ExperimentConfig()
    base = replace(base, run=replace(base.run, steps=120, save_timeseries=False, save_events=False))

    for seed in (0, 1, 2):
        cfg = replace(base, run=replace(base.run, seed=seed))
        summary_np = run_summary(cfg, prefer_numba=False)
        summary_nb = run_summary(cfg, prefer_numba=True)

        assert summary_nb["safe"] == summary_np["safe"]
        assert summary_nb["captured"] == summary_np["captured"]
        assert summary_nb["safe_frac"] == pytest.approx(summary_np["safe_frac"], abs=1e-12)
        assert summary_nb["chi"] == pytest.approx(summary_np["chi"], abs=1e-10)
        assert summary_nb["chi_local"] == pytest.approx(summary_np["chi_local"], abs=1e-10)
