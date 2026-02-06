from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, TypeVar

import numpy as np

from .geometry import rotate, unit, wrap_delta, wrap_delta_inplace

_Fn = TypeVar("_Fn", bound=Callable[..., Any])


def _identity_jit(*_args: Any, **_kwargs: Any) -> Callable[[_Fn], _Fn]:
    def deco(func: _Fn) -> _Fn:
        return func

    return deco


try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    njit = _identity_jit
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _dense_align_sep_numba(
        p: np.ndarray,
        v_dir: np.ndarray,
        periodic: bool,
        sx: float,
        sy: float,
        r2_nbr: float,
        r2_sep: float,
        eps: float,
        align_sum: np.ndarray,
        sep_sum: np.ndarray,
    ) -> None:
        m = p.shape[0]
        hx = 0.5 * sx
        hy = 0.5 * sy

        for i in range(m):
            align_sum[i, 0] = 0.0
            align_sum[i, 1] = 0.0
            sep_sum[i, 0] = 0.0
            sep_sum[i, 1] = 0.0

            xi = p[i, 0]
            yi = p[i, 1]
            for j in range(m):
                if i == j:
                    continue
                dx = p[j, 0] - xi
                dy = p[j, 1] - yi
                if periodic:
                    if dx > hx:
                        dx -= sx
                    elif dx < -hx:
                        dx += sx
                    if dy > hy:
                        dy -= sy
                    elif dy < -hy:
                        dy += sy

                d2 = dx * dx + dy * dy

                if d2 <= r2_nbr:
                    align_sum[i, 0] += v_dir[j, 0]
                    align_sum[i, 1] += v_dir[j, 1]

                if d2 <= r2_sep:
                    inv = 1.0 / (d2 if d2 > eps else eps)
                    sep_sum[i, 0] -= inv * dx
                    sep_sum[i, 1] -= inv * dy


def _numba_enabled(prefer_numba: bool | None) -> bool:
    if prefer_numba is not None:
        return bool(prefer_numba) and _NUMBA_AVAILABLE
    if not _NUMBA_AVAILABLE:
        return False
    return os.getenv("CHASECRIT_DISABLE_NUMBA", "0") != "1"


def numba_runtime_available() -> bool:
    return _NUMBA_AVAILABLE


@dataclass
class EvaderDenseWorkspace:
    """
    Scratch buffers for the dense O(N^2) evader neighbor/separation interactions.
    This avoids per-step allocations while preserving the same semantics and RNG usage.
    """

    delta: np.ndarray
    dist2: np.ndarray
    tmp: np.ndarray
    mask_n: np.ndarray
    mask_s: np.ndarray
    mask_n_f32: np.ndarray
    w_sep: np.ndarray
    prod: np.ndarray
    align_sum: np.ndarray
    sep_sum: np.ndarray

    @staticmethod
    def create(n_max: int) -> "EvaderDenseWorkspace":
        return EvaderDenseWorkspace(
            delta=np.empty((n_max, n_max, 2), dtype=np.float64),
            dist2=np.empty((n_max, n_max), dtype=np.float64),
            tmp=np.empty((n_max, n_max), dtype=np.float64),
            mask_n=np.empty((n_max, n_max), dtype=bool),
            mask_s=np.empty((n_max, n_max), dtype=bool),
            mask_n_f32=np.empty((n_max, n_max), dtype=np.float32),
            w_sep=np.empty((n_max, n_max), dtype=np.float64),
            prod=np.empty((n_max, n_max, 2), dtype=np.float64),
            align_sum=np.empty((n_max, 2), dtype=np.float64),
            sep_sum=np.empty((n_max, 2), dtype=np.float64),
        )

    @property
    def n_max(self) -> int:
        return int(self.dist2.shape[0])


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
    dense_ws: EvaderDenseWorkspace | None = None,
    prefer_numba: bool | None = None,
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
    heading = v_dir

    # Neighbor interactions.
    # For small swarms, dense pairwise vectorization is typically faster than Python-level cell loops.
    # For larger swarms, a cell-binned method avoids O(N^2).
    r2_nbr = float(r_nbr * r_nbr)
    r2_sep = float(r_sep * r_sep)
    eps = 1e-6
    use_numba = _numba_enabled(prefer_numba)

    if M <= 256 and use_numba:
        align_sum = np.empty((M, 2), dtype=np.float64)
        sep_sum = np.empty((M, 2), dtype=np.float64)
        _dense_align_sep_numba(
            p,
            v_dir,
            periodic,
            float(size[0]),
            float(size[1]),
            r2_nbr,
            r2_sep,
            eps,
            align_sum,
            sep_sum,
        )
        d_align = unit(align_sum)
        d_sep = unit(sep_sum) * sep_strength
    elif M <= 256:
        if dense_ws is not None and dense_ws.n_max >= M:
            delta = dense_ws.delta[:M, :M, :]
            np.subtract(p[None, :, :], p[:, None, :], out=delta)
            if periodic:
                wrap_delta_inplace(delta, size)

            dist2 = dense_ws.dist2[:M, :M]
            tmp = dense_ws.tmp[:M, :M]
            np.multiply(delta[:, :, 0], delta[:, :, 0], out=dist2)
            np.multiply(delta[:, :, 1], delta[:, :, 1], out=tmp)
            np.add(dist2, tmp, out=dist2)
            np.fill_diagonal(dist2, np.inf)

            mask_n = dense_ws.mask_n[:M, :M]
            np.less_equal(dist2, r2_nbr, out=mask_n)
            mask_n_f32 = dense_ws.mask_n_f32[:M, :M]
            mask_n_f32[...] = mask_n
            align_sum = dense_ws.align_sum[:M, :]
            np.matmul(mask_n_f32, v_dir, out=align_sum)

            mask_s = dense_ws.mask_s[:M, :M]
            np.less_equal(dist2, r2_sep, out=mask_s)

            # Reuse dist2 buffer as inv = 1/max(dist2, eps)
            np.maximum(dist2, eps, out=dist2)
            np.reciprocal(dist2, out=dist2)
            w_sep = dense_ws.w_sep[:M, :M]
            np.multiply(dist2, mask_s, out=w_sep, casting="unsafe")

            prod = dense_ws.prod[:M, :M, :]
            np.multiply(w_sep[:, :, None], delta, out=prod)
            sep_sum = dense_ws.sep_sum[:M, :]
            np.sum(prod, axis=1, out=sep_sum)
            sep_sum *= -1.0
        else:
            delta = p[None, :, :] - p[:, None, :]
            if periodic:
                delta = wrap_delta(delta, size)
            dist2 = np.sum(delta**2, axis=2)
            np.fill_diagonal(dist2, np.inf)

            mask_n = dist2 <= r2_nbr
            align_sum = mask_n.astype(np.float32) @ v_dir

            mask_s = dist2 <= r2_sep
            inv = 1.0 / np.maximum(dist2, eps)
            w = mask_s.astype(np.float64) * inv
            sep_sum = -(w[:, :, None] * delta).sum(axis=1)

        d_align = unit(align_sum)
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
    r2_pred = float(r_pred * r_pred)
    if pos_p.shape[0] > 0:
        delta_ap = pos_p[None, :, :] - p[:, None, :]
        if periodic:
            delta_ap = wrap_delta(delta_ap, size)
        dist2_ap = np.sum(delta_ap**2, axis=2)
        pred_mask = dist2_ap <= r2_pred
        inv_dp = 1.0 / np.maximum(dist2_ap, eps)
        w_avoid_local = pred_mask.astype(np.float64) * inv_dp
        d_avoid = unit(-(w_avoid_local[:, :, None] * delta_ap).sum(axis=1))
    else:
        d_avoid = np.zeros((M, 2), dtype=np.float64)

    # Goal towards nearest detectable active zone (K is small).
    r2_detect = float(r_detect * r_detect)
    active_z = np.flatnonzero(zone_active)
    has_goal = np.zeros((M,), dtype=bool)
    d_goal = np.zeros((M, 2), dtype=np.float64)
    if active_z.size > 0:
        zpos = pos_z[active_z]
        delta_az = zpos[None, :, :] - p[:, None, :]
        if periodic:
            delta_az = wrap_delta(delta_az, size)
        dist2_az = np.sum(delta_az**2, axis=2)
        detectable = dist2_az <= r2_detect
        dist2_masked = dist2_az.copy()
        dist2_masked[~detectable] = np.inf
        k = np.argmin(dist2_masked, axis=1)
        best = dist2_masked[np.arange(M), k]
        has_goal = np.isfinite(best)
        best_delta = delta_az[np.arange(M), k]
        d_goal = unit(best_delta)

    # Explore: keep heading or random.
    near_zero = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) < 1e-12
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
