from __future__ import annotations

import numpy as np

from .geometry import rotate, unit, wrap_delta


def evader_step(
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
    """
    Returns new velocities for evaders (positions updated outside).
    All arrays are full-size (N_e,2) etc; alive_mask marks active evaders.
    """
    N = pos_e.shape[0]
    size = world_size
    alive_idx = np.flatnonzero(alive_mask)
    if alive_idx.size == 0:
        return vel_e.copy()

    p_e = pos_e
    v_e_cur = vel_e

    # Pairwise deltas among evaders for neighbor/separation.
    delta_ee = p_e[None, :, :] - p_e[:, None, :]
    if periodic:
        delta_ee = wrap_delta(delta_ee, size)
    dist2_ee = np.sum(delta_ee**2, axis=2)
    np.fill_diagonal(dist2_ee, np.inf)

    # Neighbors (for all i, j)
    nbr_mask = dist2_ee <= (r_nbr * r_nbr)
    sep_mask = dist2_ee <= (r_sep * r_sep)

    # Alignment: average neighbor velocity direction
    v_dir = unit(v_e_cur)
    v_dir[~alive_mask] = 0.0
    align_sum = np.einsum("ij,jk->ik", nbr_mask.astype(np.float32), v_dir)
    d_align = unit(align_sum)

    # Separation: repulse from close neighbors
    inv_d = 1.0 / np.maximum(dist2_ee, 1e-6)
    repulse = -(delta_ee * inv_d[:, :, None])  # away from neighbors
    repulse[~sep_mask] = 0.0
    repulse_sum = repulse.sum(axis=1)
    d_sep = unit(repulse_sum) * sep_strength

    # Avoid pursuers
    if pos_p.shape[0] > 0:
        delta_ep = pos_p[None, :, :] - p_e[:, None, :]
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
        d_avoid = np.zeros_like(p_e)

    # Goal towards nearest detectable active safe zone
    active_z = np.flatnonzero(zone_active)
    d_goal = np.zeros_like(p_e)
    has_goal = np.zeros((N,), dtype=bool)
    if active_z.size > 0:
        zpos = pos_z[active_z]
        delta_ez = zpos[None, :, :] - p_e[:, None, :]
        if periodic:
            delta_ez = wrap_delta(delta_ez, size)
        dist2_ez = np.sum(delta_ez**2, axis=2)
        detectable = dist2_ez <= (r_detect * r_detect)
        # Pick nearest detectable; if none detectable -> no goal
        dist2_masked = np.where(detectable, dist2_ez, np.inf)
        k = np.argmin(dist2_masked, axis=1)
        best = dist2_masked[np.arange(N), k]
        has_goal = np.isfinite(best)
        best_delta = delta_ez[np.arange(N), k]
        d_goal = unit(best_delta)

    # Explore direction: keep heading, or random if near-zero.
    heading = unit(v_e_cur)
    near_zero = np.linalg.norm(v_e_cur, axis=1) < 1e-6
    rand_dir = unit(rng.normal(size=(N, 2)))
    d_explore = heading.copy()
    d_explore[near_zero] = rand_dir[near_zero]

    # Blend
    d = (
        w_align * d_align
        + w_avoid * d_avoid
        + w_goal * (d_goal * has_goal[:, None])
        + w_explore * (d_explore * (~has_goal)[:, None])
        + d_sep
    )
    d = unit(d)

    # Noise (only for alive)
    if angle_noise > 0:
        angles = rng.uniform(-angle_noise, angle_noise, size=(N,))
        d = rotate(d, angles)
        d = unit(d)

    v_next = v_e_cur * (1.0 - inertia) + (v_e * d) * inertia
    v_next[~alive_mask] = 0.0
    return v_next


def pursuer_step_p0_nearest(
    *,
    pos_p: np.ndarray,
    pos_e: np.ndarray,
    alive_mask: np.ndarray,
    world_size: np.ndarray,
    periodic: bool,
    v_p: float,
) -> np.ndarray:
    if pos_p.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if alive_mask.sum() == 0:
        return np.zeros_like(pos_p)
    size = world_size
    alive_pos = pos_e[alive_mask]

    # For each pursuer, chase nearest alive evader
    delta = alive_pos[None, :, :] - pos_p[:, None, :]
    if periodic:
        delta = wrap_delta(delta, size)
    dist2 = np.sum(delta**2, axis=2)
    idx = np.argmin(dist2, axis=1)
    d = unit(delta[np.arange(pos_p.shape[0]), idx])
    return v_p * d

