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
    alive_idx = np.flatnonzero(alive_mask)
    if alive_idx.size == 0:
        out = vel_e.copy()
        out[~alive_mask] = 0.0
        return out

    size = world_size
    p = pos_e[alive_idx]
    v = vel_e[alive_idx]
    M = p.shape[0]

    v_dir = unit(v)

    # Neighbor interactions.
    # For small swarms, dense pairwise vectorization is typically faster than Python-level cell loops.
    # For larger swarms, a cell-binned method avoids O(N^2).
    r2_nbr = float(r_nbr * r_nbr)
    r2_sep = float(r_sep * r_sep)
    eps = 1e-6

    if M <= 256:
        delta = p[None, :, :] - p[:, None, :]
        if periodic:
            delta = wrap_delta(delta, size)
        dist2 = np.sum(delta**2, axis=2)
        np.fill_diagonal(dist2, np.inf)

        mask_n = dist2 <= r2_nbr
        align_sum = mask_n.astype(np.float32) @ v_dir
        d_align = unit(align_sum)

        mask_s = dist2 <= r2_sep
        inv = 1.0 / np.maximum(dist2, eps)
        w = mask_s.astype(np.float64) * inv
        sep_sum = -(w[:, :, None] * delta).sum(axis=1)
        d_sep = unit(sep_sum) * sep_strength
    else:
        # Cell-binned interactions (Moore neighborhood, 9 cells).
        cell_size = max(r_nbr, r_sep)
        nx = max(1, int(np.ceil(float(size[0]) / cell_size)))
        ny = max(1, int(np.ceil(float(size[1]) / cell_size)))

        cx = np.floor(p[:, 0] / cell_size).astype(np.int32)
        cy = np.floor(p[:, 1] / cell_size).astype(np.int32)
        if periodic:
            cx %= nx
            cy %= ny
        else:
            cx = np.clip(cx, 0, nx - 1)
            cy = np.clip(cy, 0, ny - 1)

        cid = cx + nx * cy
        order = np.argsort(cid, kind="mergesort")
        cid_sorted = cid[order]
        unique_cells, start_idx, counts = np.unique(cid_sorted, return_index=True, return_counts=True)

        cell_slices: dict[int, tuple[int, int]] = {}
        for c, s, k in zip(unique_cells.tolist(), start_idx.tolist(), counts.tolist(), strict=True):
            cell_slices[int(c)] = (int(s), int(s + k))

        align_sum = np.zeros((M, 2), dtype=np.float64)
        sep_sum = np.zeros((M, 2), dtype=np.float64)

        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

        for c in unique_cells.tolist():
            c = int(c)
            sA, eA = cell_slices[c]
            IA = order[sA:eA]
            if IA.size == 0:
                continue

            cx0 = c % nx
            cy0 = c // nx

            neighbor_ids: list[int] = []
            seen: set[int] = set()
            for dx, dy in neighbor_offsets:
                x = cx0 + dx
                y = cy0 + dy
                if periodic:
                    x %= nx
                    y %= ny
                else:
                    if x < 0 or x >= nx or y < 0 or y >= ny:
                        continue
                cn = int(x + nx * y)
                if cn in cell_slices and cn not in seen:
                    seen.add(cn)
                    neighbor_ids.append(cn)

            for cn in neighbor_ids:
                if cn < c:
                    continue
                sB, eB = cell_slices[cn]
                IB = order[sB:eB]
                if IB.size == 0:
                    continue

                pA = p[IA]
                pB = p[IB]
                delta = pB[None, :, :] - pA[:, None, :]
                if periodic:
                    delta = wrap_delta(delta, size)
                dist2 = np.sum(delta**2, axis=2)

                if cn == c:
                    np.fill_diagonal(dist2, np.inf)

                mask_n = dist2 <= r2_nbr
                if mask_n.any():
                    align_sum[IA] += mask_n @ v_dir[IB]
                    if cn != c:
                        align_sum[IB] += mask_n.T @ v_dir[IA]

                mask_s = dist2 <= r2_sep
                if mask_s.any():
                    inv = 1.0 / np.maximum(dist2, eps)
                    w = mask_s.astype(np.float64) * inv
                    repA = -(w[:, :, None] * delta).sum(axis=1)
                    sep_sum[IA] += repA
                    if cn != c:
                        repB = (w[:, :, None] * delta).sum(axis=0)
                        sep_sum[IB] += repB

        d_align = unit(align_sum)
        d_sep = unit(sep_sum) * sep_strength

    # Avoid pursuers (M x Np, Np is small).
    if pos_p.shape[0] > 0:
        delta_ap = pos_p[None, :, :] - p[:, None, :]
        if periodic:
            delta_ap = wrap_delta(delta_ap, size)
        dist2_ap = np.sum(delta_ap**2, axis=2)
        pred_mask = dist2_ap <= (r_pred * r_pred)
        inv_dp = 1.0 / np.maximum(dist2_ap, eps)
        away = -(delta_ap * inv_dp[:, :, None])
        away[~pred_mask] = 0.0
        d_avoid = unit(away.sum(axis=1))
    else:
        d_avoid = np.zeros((M, 2), dtype=np.float64)

    # Goal towards nearest detectable active zone (K is small).
    active_z = np.flatnonzero(zone_active)
    has_goal = np.zeros((M,), dtype=bool)
    d_goal = np.zeros((M, 2), dtype=np.float64)
    if active_z.size > 0:
        zpos = pos_z[active_z]
        delta_az = zpos[None, :, :] - p[:, None, :]
        if periodic:
            delta_az = wrap_delta(delta_az, size)
        dist2_az = np.sum(delta_az**2, axis=2)
        detectable = dist2_az <= (r_detect * r_detect)
        dist2_masked = np.where(detectable, dist2_az, np.inf)
        k = np.argmin(dist2_masked, axis=1)
        best = dist2_masked[np.arange(M), k]
        has_goal = np.isfinite(best)
        best_delta = delta_az[np.arange(M), k]
        d_goal = unit(best_delta)

    # Explore: keep heading or random.
    heading = unit(v)
    near_zero = np.linalg.norm(v, axis=1) < 1e-6
    rand_dir = unit(rng.normal(size=(M, 2)))
    d_explore = heading
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
