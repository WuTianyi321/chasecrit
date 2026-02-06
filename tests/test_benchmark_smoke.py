from __future__ import annotations

from dataclasses import replace

import pytest

from chasecrit.bench import run_small_benchmark
from chasecrit.config import ExperimentConfig
from chasecrit.policies import numba_runtime_available


def test_small_benchmark_smoke() -> None:
    cfg = ExperimentConfig()
    cfg = replace(cfg, run=replace(cfg.run, steps=32, save_timeseries=False, save_events=False))
    result = run_small_benchmark(cfg, iterations=16, steps=32, repeats=1, seed=0)

    assert "evader_step_speedup_x" in result
    assert "simulation_speedup_x" in result
    assert result["evader_step_numpy_mean_s"] > 0.0
    assert result["evader_step_numba_mean_s"] > 0.0
    assert result["simulation_numpy_mean_s"] > 0.0
    assert result["simulation_numba_mean_s"] > 0.0


@pytest.mark.skipif(not numba_runtime_available(), reason="numba not available")
def test_small_benchmark_numba_speedup_sanity() -> None:
    cfg = ExperimentConfig()
    cfg = replace(cfg, run=replace(cfg.run, steps=96, save_timeseries=False, save_events=False))
    result = run_small_benchmark(cfg, iterations=64, steps=96, repeats=1, seed=123)

    assert result["evader_step_speedup_x"] > 1.05
    assert result["simulation_speedup_x"] > 1.02
