from __future__ import annotations

import numpy as np

from chasecrit.policies import pursuer_step_p0_nearest, pursuer_step_p1_intercept


def test_p1_intercept_leads_lateral_target() -> None:
    pos_p = np.array([[0.0, 0.0]], dtype=np.float64)
    pos_e = np.array([[10.0, 0.0]], dtype=np.float64)
    vel_e = np.array([[0.0, 1.0]], dtype=np.float64)
    alive = np.array([True], dtype=bool)
    world = np.array([100.0, 100.0], dtype=np.float64)

    v0 = pursuer_step_p0_nearest(
        pos_p=pos_p,
        pos_e=pos_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
    )
    v1 = pursuer_step_p1_intercept(
        pos_p=pos_p,
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
    )

    assert v0[0, 1] == 0.0
    assert v1[0, 1] > 0.0


def test_p1_intercept_falls_back_to_nearest_when_no_solution() -> None:
    pos_p = np.array([[0.0, 0.0]], dtype=np.float64)
    pos_e = np.array([[10.0, 0.0]], dtype=np.float64)
    vel_e = np.array([[2.0, 0.0]], dtype=np.float64)  # faster target moving away
    alive = np.array([True], dtype=bool)
    world = np.array([100.0, 100.0], dtype=np.float64)

    v0 = pursuer_step_p0_nearest(
        pos_p=pos_p,
        pos_e=pos_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=1.0,
    )
    v1 = pursuer_step_p1_intercept(
        pos_p=pos_p,
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=1.0,
    )

    np.testing.assert_allclose(v1, v0, atol=1e-12, rtol=0.0)


def test_p1_intercept_gain_zero_matches_nearest() -> None:
    pos_p = np.array([[0.0, 0.0]], dtype=np.float64)
    pos_e = np.array([[10.0, 0.0]], dtype=np.float64)
    vel_e = np.array([[0.0, 1.0]], dtype=np.float64)
    alive = np.array([True], dtype=bool)
    world = np.array([100.0, 100.0], dtype=np.float64)

    v0 = pursuer_step_p0_nearest(
        pos_p=pos_p,
        pos_e=pos_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
    )
    v1_gain0 = pursuer_step_p1_intercept(
        pos_p=pos_p,
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
        intercept_gain=0.0,
    )

    np.testing.assert_allclose(v1_gain0, v0, atol=1e-12, rtol=0.0)


def test_p1_intercept_short_horizon_reduces_lateral_lead() -> None:
    pos_p = np.array([[0.0, 0.0]], dtype=np.float64)
    pos_e = np.array([[10.0, 0.0]], dtype=np.float64)
    vel_e = np.array([[0.0, 1.0]], dtype=np.float64)
    alive = np.array([True], dtype=bool)
    world = np.array([100.0, 100.0], dtype=np.float64)

    v_full = pursuer_step_p1_intercept(
        pos_p=pos_p,
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
        intercept_gain=1.0,
        intercept_tmax=0.0,
    )
    v_short = pursuer_step_p1_intercept(
        pos_p=pos_p,
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        world_size=world,
        periodic=False,
        v_p=2.0,
        intercept_gain=1.0,
        intercept_tmax=0.5,
    )

    assert v_full[0, 1] > v_short[0, 1] > 0.0
