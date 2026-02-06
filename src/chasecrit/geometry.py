from __future__ import annotations

import numpy as np


def wrap_delta(delta: np.ndarray, size: np.ndarray) -> np.ndarray:
    """
    Map displacement to the shortest equivalent displacement under periodic boundary.
    delta: (..., 2), size: (2,)
    """
    return (delta + 0.5 * size) % size - 0.5 * size


def wrap_delta_inplace(delta: np.ndarray, size: np.ndarray) -> None:
    """
    In-place version of wrap_delta for periodic boundaries.
    delta is modified in-place.
    """
    Lx = float(size[0])
    Ly = float(size[1])

    dx = delta[..., 0]
    dy = delta[..., 1]

    hx = 0.5 * Lx
    hy = 0.5 * Ly

    np.add(dx, hx, out=dx)
    np.remainder(dx, Lx, out=dx)
    np.subtract(dx, hx, out=dx)

    np.add(dy, hy, out=dy)
    np.remainder(dy, Ly, out=dy)
    np.subtract(dy, hy, out=dy)


def apply_periodic(pos: np.ndarray, size: np.ndarray) -> np.ndarray:
    return pos % size


def apply_reflecting(pos: np.ndarray, vel: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reflect positions/velocities at [0,size] walls.
    Works for one-step integration; multiple crossings are handled by mod-like folding.
    """
    p = pos.copy()
    v = vel.copy()
    for axis in (0, 1):
        L = float(size[axis])
        x = p[:, axis]
        x_mod = np.mod(x, 2 * L)
        reflected = x_mod > L
        x_mod[reflected] = 2 * L - x_mod[reflected]
        p[:, axis] = x_mod

        # Flip velocity sign on the reflected half (odd number of wall hits).
        v[reflected, axis] *= -1.0
    return p, v


def unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(n, eps)


def rotate(vec: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Rotate 2D vectors by angles (radians). vec: (N,2), angles: (N,) or scalar.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    x = vec[:, 0] * c - vec[:, 1] * s
    y = vec[:, 0] * s + vec[:, 1] * c
    return np.stack([x, y], axis=1)
