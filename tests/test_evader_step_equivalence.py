from __future__ import annotations

import numpy as np

from chasecrit.policies import EvaderDenseWorkspace, evader_step
from chasecrit.geometry import rotate, unit, wrap_delta


def _reference_evader_step_dense(
    *,
    pos_e: np.ndarray,
    vel_e: np.ndarray,
    alive_mask: np.ndarray,
    pos_p: np.ndarray,
    pos_z: np.ndarray,
    zone_active: np.ndarray,
    world_size: np.ndarray,
    periodic: bool,
    rng: np.random.Generator,
    v_e: float,
    inertia: float,
    r_nbr: float,
    r_pred: float,
    r_detect: float,
    r_sep: float,
    w_align: float,
    w_goal: float,
    w_avoid: float,
    w_explore: float,
    sep_strength: float,
    angle_noise: float,
) -> np.ndarray:
    # Dense version mirroring the *current* semantics:
    # only alive evaders participate in neighbor/separation interactions.
    N = pos_e.shape[0]
    alive_idx = np.flatnonzero(alive_mask)
    if alive_idx.size == 0:
        out = vel_e.copy()
        out[~alive_mask] = 0.0
        return out

    p = pos_e[alive_idx]
    v = vel_e[alive_idx]
    M = p.shape[0]
    size = world_size

    delta = p[None, :, :] - p[:, None, :]
    if periodic:
        delta = wrap_delta(delta, size)
    dist2 = np.sum(delta**2, axis=2)
    np.fill_diagonal(dist2, np.inf)

    nbr_mask = dist2 <= (r_nbr * r_nbr)
    sep_mask = dist2 <= (r_sep * r_sep)

    v_dir = unit(v)
    align_sum = nbr_mask.astype(np.float32) @ v_dir
    d_align = unit(align_sum)

    inv_d = 1.0 / np.maximum(dist2, 1e-6)
    repulse = -(delta * inv_d[:, :, None])
    repulse[~sep_mask] = 0.0
    repulse_sum = repulse.sum(axis=1)
    d_sep = unit(repulse_sum) * sep_strength

    if pos_p.shape[0] > 0:
        delta_ep = pos_p[None, :, :] - p[:, None, :]
        if periodic:
            delta_ep = wrap_delta(delta_ep, size)
        dist2_ep = np.sum(delta_ep**2, axis=2)
        pred_mask = dist2_ep <= (r_pred * r_pred)
        inv_dp = 1.0 / np.maximum(dist2_ep, 1e-6)
        away = -(delta_ep * inv_dp[:, :, None])
        away[~pred_mask] = 0.0
        avoid_sum = away.sum(axis=1)
        d_avoid = unit(avoid_sum)
    else:
        d_avoid = np.zeros_like(p)

    active_z = np.flatnonzero(zone_active)
    d_goal = np.zeros_like(p)
    has_goal = np.zeros((M,), dtype=bool)
    if active_z.size > 0:
        zpos = pos_z[active_z]
        delta_ez = zpos[None, :, :] - p[:, None, :]
        if periodic:
            delta_ez = wrap_delta(delta_ez, size)
        dist2_ez = np.sum(delta_ez**2, axis=2)
        detectable = dist2_ez <= (r_detect * r_detect)
        dist2_masked = np.where(detectable, dist2_ez, np.inf)
        k = np.argmin(dist2_masked, axis=1)
        best = dist2_masked[np.arange(M), k]
        has_goal = np.isfinite(best)
        best_delta = delta_ez[np.arange(M), k]
        d_goal = unit(best_delta)

    heading = unit(v)
    near_zero = np.linalg.norm(v, axis=1) < 1e-6
    rand_dir = unit(rng.normal(size=(M, 2)))
    d_explore = heading.copy()
    d_explore[near_zero] = rand_dir[near_zero]

    d = (
        w_align * d_align
        + w_avoid * d_avoid
        + w_goal * (d_goal * has_goal[:, None])
        + w_explore * (d_explore * (~has_goal)[:, None])
        + d_sep
    )
    d = unit(d)

    if angle_noise > 0:
        angles = rng.uniform(-angle_noise, angle_noise, size=(M,))
        d = unit(rotate(d, angles))

    v_next_alive = v * (1.0 - inertia) + (v_e * d) * inertia

    out = vel_e.copy()
    out[alive_idx] = v_next_alive
    out[~alive_mask] = 0.0
    return out


def test_evader_step_matches_dense_reference() -> None:
    rng_init = np.random.default_rng(999)
    rng0 = np.random.default_rng(123)
    rng1 = np.random.default_rng(123)

    N = 64
    world_size = np.array([100.0, 100.0], dtype=np.float64)
    pos_e = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(N, 2)).astype(np.float64)
    vel_e = unit(rng_init.normal(size=(N, 2))).astype(np.float64)
    alive = rng_init.random(size=(N,)) > 0.2
    pos_p = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(2, 2)).astype(np.float64)
    pos_z = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(3, 2)).astype(np.float64)
    zone_active = np.array([True, False, True])

    params = dict(
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        pos_p=pos_p,
        pos_z=pos_z,
        zone_active=zone_active,
        world_size=world_size,
        periodic=True,
        v_e=1.0,
        inertia=0.35,
        r_nbr=6.0,
        r_pred=10.0,
        r_detect=4.0,
        r_sep=1.25,
        w_align=0.6,
        w_goal=1.2,
        w_avoid=1.5,
        w_explore=0.4,
        sep_strength=1.0,
        angle_noise=0.12,
    )

    ref = _reference_evader_step_dense(rng=rng0, **params)
    got = evader_step(rng=rng1, **params)

    assert np.allclose(got, ref, rtol=0.0, atol=1e-10)


def test_evader_step_dense_workspace_matches_no_workspace() -> None:
    rng_init = np.random.default_rng(2026)
    rng0 = np.random.default_rng(777)
    rng1 = np.random.default_rng(777)

    N = 128
    world_size = np.array([200.0, 200.0], dtype=np.float64)
    pos_e = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(N, 2)).astype(np.float64)
    vel_e = unit(rng_init.normal(size=(N, 2))).astype(np.float64)
    alive = rng_init.random(size=(N,)) > 0.15
    pos_p = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(2, 2)).astype(np.float64)
    pos_z = rng_init.uniform(low=[0.0, 0.0], high=world_size, size=(4, 2)).astype(np.float64)
    zone_active = rng_init.random(size=(4,)) > 0.3

    params = dict(
        pos_e=pos_e,
        vel_e=vel_e,
        alive_mask=alive,
        pos_p=pos_p,
        pos_z=pos_z,
        zone_active=zone_active,
        world_size=world_size,
        periodic=True,
        v_e=1.0,
        inertia=0.35,
        r_nbr=6.0,
        r_pred=10.0,
        r_detect=4.0,
        r_sep=1.25,
        w_align=0.6,
        w_goal=1.2,
        w_avoid=1.5,
        w_explore=0.4,
        sep_strength=1.0,
        angle_noise=0.12,
    )

    baseline = evader_step(rng=rng0, **params)
    ws = EvaderDenseWorkspace.create(N)
    got = evader_step(rng=rng1, dense_ws=ws, **params)

    assert np.allclose(got, baseline, rtol=0.0, atol=1e-10)
