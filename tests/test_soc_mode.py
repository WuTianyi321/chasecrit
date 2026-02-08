from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from chasecrit.config import ExperimentConfig
from chasecrit.policies import EvaderSocState, evader_step
from chasecrit.sim import run_summary


def _step_inputs() -> dict[str, object]:
    pos_e = np.array(
        [
            [0.0, 0.0],
            [0.6, 0.0],
            [1.2, 0.0],
            [1.8, 0.0],
        ],
        dtype=np.float64,
    )
    vel_e = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    alive_mask = np.array([True, True, True, True], dtype=bool)
    pos_p = np.array([[0.2, 0.0]], dtype=np.float64)
    pos_z = np.zeros((0, 2), dtype=np.float64)
    zone_active = np.zeros((0,), dtype=bool)
    world_size = np.array([20.0, 20.0], dtype=np.float64)
    return {
        "pos_e": pos_e,
        "vel_e": vel_e,
        "alive_mask": alive_mask,
        "pos_p": pos_p,
        "pos_z": pos_z,
        "zone_active": zone_active,
        "world_size": world_size,
        "periodic": False,
        "v_e": 1.0,
        "inertia": 1.0,
        "r_nbr": 2.0,
        "r_pred": 5.0,
        "r_detect": 1.0,
        "r_sep": 0.1,
        "w_align": 0.6,
        "w_goal": 0.0,
        "w_avoid": 1.0,
        "w_explore": 0.0,
        "sep_strength": 1.0,
        "angle_noise": 0.0,
        "align_control_mode": "legacy",
    }


def test_soc_disabled_keeps_evader_step_identical() -> None:
    kwargs = _step_inputs()

    v_ref = evader_step(rng=np.random.default_rng(11), soc_enabled=False, soc_state=None, **kwargs)

    soc_state = EvaderSocState.create(n_evaders=4, init_align_share=0.6)
    soc_state.stress[:] = np.array([2.0, 1.0, 0.5, 0.0], dtype=np.float64)
    soc_state.align_share[:] = np.array([0.2, 0.4, 0.8, 0.9], dtype=np.float64)
    v_with_state = evader_step(rng=np.random.default_rng(11), soc_enabled=False, soc_state=soc_state, **kwargs)

    np.testing.assert_allclose(v_ref, v_with_state, atol=0.0, rtol=0.0)
    assert soc_state.last_topple_count == 0


def test_soc_enabled_triggers_topple_and_updates_state() -> None:
    kwargs = _step_inputs()
    soc_state = EvaderSocState.create(n_evaders=4, init_align_share=0.6)

    _ = evader_step(
        rng=np.random.default_rng(0),
        soc_enabled=True,
        soc_state=soc_state,
        soc_threshold=0.02,
        soc_surprise_gain=0.6,
        soc_threat_gain=0.8,
        soc_stress_decay=0.05,
        soc_release=0.3,
        soc_neighbor_coupling=0.25,
        soc_align_drop=0.35,
        soc_align_relax=0.02,
        soc_align_min=0.05,
        soc_align_max=0.95,
        soc_topple_noise=0.0,
        **kwargs,
    )

    assert soc_state.last_topple_count > 0
    assert np.any(soc_state.last_toppled)
    assert np.all(soc_state.stress >= 0.0)
    assert np.all(soc_state.align_share >= 0.05 - 1e-12)
    assert np.all(soc_state.align_share <= 0.95 + 1e-12)
    assert np.any(soc_state.align_share < 0.6)


def test_soc_parameter_changes_do_not_affect_run_when_disabled() -> None:
    cfg = ExperimentConfig()
    cfg = replace(cfg, run=replace(cfg.run, steps=120, seed=3, save_timeseries=False, save_events=False))
    s_ref = run_summary(cfg, prefer_numba=False)

    cfg_soc_params = replace(
        cfg,
        evaders=replace(
            cfg.evaders,
            soc_enabled=False,
            soc_threshold=0.01,
            soc_surprise_gain=1.2,
            soc_threat_gain=1.2,
            soc_stress_decay=0.2,
            soc_release=0.8,
            soc_neighbor_coupling=0.5,
            soc_align_drop=0.5,
            soc_align_relax=0.3,
            soc_topple_noise=0.5,
        ),
    )
    s_new = run_summary(cfg_soc_params, prefer_numba=False)

    assert s_new["safe"] == s_ref["safe"]
    assert s_new["captured"] == s_ref["captured"]
    assert s_new["steps_recorded"] == s_ref["steps_recorded"]
    assert s_new["safe_frac"] == pytest.approx(s_ref["safe_frac"], abs=1e-12)
    assert s_new["chi"] == pytest.approx(s_ref["chi"], abs=1e-12)


def test_soc_v3_ignores_w_align_relaxation_when_state_fixed() -> None:
    kwargs_a = _step_inputs()
    kwargs_b = _step_inputs()
    kwargs_a["w_align"] = 0.1
    kwargs_b["w_align"] = 0.9
    soc_state_a = EvaderSocState.create(n_evaders=4, init_align_share=0.6, heading_bins=24)
    soc_state_b = EvaderSocState.create(n_evaders=4, init_align_share=0.6, heading_bins=24)

    v_a = evader_step(
        rng=np.random.default_rng(7),
        soc_enabled=True,
        soc_state=soc_state_a,
        soc_mode="v3",
        soc_relax_to_w_align=False,
        soc_heading_bins=24,
        **kwargs_a,
    )
    v_b = evader_step(
        rng=np.random.default_rng(7),
        soc_enabled=True,
        soc_state=soc_state_b,
        soc_mode="v3",
        soc_relax_to_w_align=False,
        soc_heading_bins=24,
        **kwargs_b,
    )

    np.testing.assert_allclose(v_a, v_b, atol=1e-12, rtol=0.0)


def test_soc_v3_entropy_drive_updates_entropy_statistics() -> None:
    kwargs = _step_inputs()
    soc_state = EvaderSocState.create(n_evaders=4, init_align_share=0.6, heading_bins=24)

    _ = evader_step(
        rng=np.random.default_rng(9),
        soc_enabled=True,
        soc_state=soc_state,
        soc_mode="v3",
        soc_entropy_gain=1.0,
        soc_heading_bins=24,
        soc_heading_decay=0.02,
        soc_entropy_ema_alpha=0.2,
        **kwargs,
    )
    _ = evader_step(
        rng=np.random.default_rng(10),
        soc_enabled=True,
        soc_state=soc_state,
        soc_mode="v3",
        soc_entropy_gain=1.0,
        soc_heading_bins=24,
        soc_heading_decay=0.02,
        soc_entropy_ema_alpha=0.2,
        **kwargs,
    )

    assert soc_state.last_pred_entropy_mean >= 0.0
    assert soc_state.last_pred_entropy_var >= 0.0
    assert soc_state.last_entropy_fluct_mean >= 0.0


def test_soc_v4_varh_target_controls_align_share_direction() -> None:
    kwargs = _step_inputs()

    def run_target(target: float) -> tuple[float, float]:
        state = EvaderSocState.create(n_evaders=4, init_align_share=0.6, heading_bins=24)
        for seed in (21, 22, 23):
            _ = evader_step(
                rng=np.random.default_rng(seed),
                soc_enabled=True,
                soc_state=state,
                soc_mode="v4_varh",
                soc_heading_bins=24,
                soc_heading_decay=0.02,
                soc_entropy_ema_alpha=0.2,
                soc_varh_target=target,
                soc_varh_gain=4.0,
                soc_varh_deadband=0.0,
                soc_relax_to_w_align=False,
                soc_surprise_gain=0.0,
                soc_threat_gain=0.0,
                soc_entropy_gain=0.0,
                **kwargs,
            )
        return float(np.mean(state.align_share)), float(state.last_pred_entropy_var)

    mean_align_high_target, varh_high_target = run_target(0.12)
    mean_align_low_target, varh_low_target = run_target(0.0)

    assert varh_high_target >= 0.0
    assert varh_low_target >= 0.0
    assert mean_align_high_target < mean_align_low_target
