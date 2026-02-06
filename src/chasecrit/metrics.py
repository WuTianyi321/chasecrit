from __future__ import annotations

import numpy as np

from .geometry import unit, wrap_delta

def polarization(vel: np.ndarray, alive_mask: np.ndarray) -> float:
    alive = vel[alive_mask]
    if alive.shape[0] == 0:
        return 0.0
    vsum = alive.sum(axis=0)
    denom = np.linalg.norm(alive, axis=1).sum()
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(vsum) / denom)


def correlation_length_fluctuation(
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    alive_mask: np.ndarray,
    world_size: np.ndarray,
    periodic: bool,
    bins: int = 20,
    r_max: float | None = None,
) -> float:
    """
    Estimate a correlation length from velocity-direction *fluctuations* at the final state.
    This is a lightweight proxy for spatial correlations: larger values indicate longer-range alignment correlations.

    Method:
    - Normalize velocities to direction v̂.
    - Compute mean direction m = mean(v̂).
    - Fluctuations u_i = v̂_i - m.
    - For each pair (i<j), compute dot(u_i, u_j) and distance r_ij.
    - Bin by r, compute C(r)=mean(dot(u_i,u_j)) per bin.
    - Use positive part of C(r) to compute an integral correlation length:
        ξ = sum(r * C_+(r)) / sum(C_+(r))
    """
    p = pos[alive_mask]
    v = vel[alive_mask]
    M = p.shape[0]
    if M < 4:
        return 0.0

    vhat = unit(v)
    m = vhat.mean(axis=0, keepdims=True)
    u = vhat - m

    delta = p[None, :, :] - p[:, None, :]
    if periodic:
        delta = wrap_delta(delta, world_size)
    dist = np.sqrt(np.sum(delta**2, axis=2))
    dot = u @ u.T

    iu = np.triu_indices(M, k=1)
    rij = dist[iu]
    cij = dot[iu]

    if r_max is None:
        r_max = float(min(world_size[0], world_size[1]) / 2.0)
    if r_max <= 1e-9:
        return 0.0

    edges = np.linspace(0.0, r_max, bins + 1, dtype=np.float64)
    idx = np.searchsorted(edges, rij, side="right") - 1
    valid = (idx >= 0) & (idx < bins)
    if not valid.any():
        return 0.0

    idx = idx[valid]
    cij = cij[valid]
    # mean per bin
    sums = np.bincount(idx, weights=cij, minlength=bins).astype(np.float64)
    cnts = np.bincount(idx, minlength=bins).astype(np.float64)
    C = np.divide(sums, np.maximum(cnts, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Use a decay-length estimate to avoid spurious large ξ when correlations are ~0.
    nonempty = cnts > 0
    if not nonempty.any():
        return 0.0
    first = int(np.flatnonzero(nonempty)[0])
    C0 = float(C[first])
    if C0 <= 1e-6:
        return 0.0

    target = C0 / float(np.e)
    for i in range(first, bins):
        if not nonempty[i]:
            continue
        if float(C[i]) <= target:
            return float(centers[i])
    return float(centers[-1])


def local_polarization_stats(
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    alive_mask: np.ndarray,
    world_size: np.ndarray,
    periodic: bool,
    neighbor_radius: float,
) -> tuple[float, float]:
    """
    Compute mean/variance of local polarization P_local(i) over alive evaders.
    P_local(i) = |sum_{j in N_i} v̂_j| / |N_i|, where N_i includes i and neighbors within neighbor_radius.
    """
    p = pos[alive_mask]
    v = vel[alive_mask]
    M = p.shape[0]
    if M == 0:
        return 0.0, 0.0

    vhat = unit(v)
    delta = p[None, :, :] - p[:, None, :]
    if periodic:
        delta = wrap_delta(delta, world_size)
    dist2 = np.sum(delta**2, axis=2)

    nbr = dist2 <= (neighbor_radius * neighbor_radius)
    # include self
    np.fill_diagonal(nbr, True)

    k = nbr.sum(axis=1).astype(np.float64)
    sums = nbr.astype(np.float32) @ vhat
    P_local = np.linalg.norm(sums, axis=1) / np.maximum(k, 1.0)

    return float(P_local.mean()), float(P_local.var())


def component_count(
    *,
    pos: np.ndarray,
    alive_mask: np.ndarray,
    world_size: np.ndarray,
    periodic: bool,
    radius: float,
) -> int:
    """
    Count connected components in the evader graph using a distance threshold (radius).
    Uses O(M^2) dense distances for M alive evaders (intended for small/medium M).
    """
    p = pos[alive_mask]
    M = p.shape[0]
    if M == 0:
        return 0
    if M == 1:
        return 1

    delta = p[None, :, :] - p[:, None, :]
    if periodic:
        delta = wrap_delta(delta, world_size)
    dist2 = np.sum(delta**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    adj = dist2 <= (radius * radius)

    seen = np.zeros((M,), dtype=bool)
    comps = 0
    for i in range(M):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            u_idx = stack.pop()
            nbrs = np.flatnonzero(adj[u_idx] & (~seen))
            if nbrs.size:
                seen[nbrs] = True
                stack.extend(nbrs.tolist())
    return int(comps)
