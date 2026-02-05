from __future__ import annotations

import numpy as np


def polarization(vel: np.ndarray, alive_mask: np.ndarray) -> float:
    alive = vel[alive_mask]
    if alive.shape[0] == 0:
        return 0.0
    vsum = alive.sum(axis=0)
    denom = np.linalg.norm(alive, axis=1).sum()
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(vsum) / denom)

