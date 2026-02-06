from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .config import SafeZoneConfig, nominal_zone_capacity
from .geometry import apply_periodic_inplace, unit


@dataclass
class SafeZones:
    pos: np.ndarray  # (K,2)
    vel: np.ndarray  # (K,2)
    cap: np.ndarray  # (K,)
    occ: np.ndarray  # (K,)
    active: np.ndarray  # (K,)
    active_total: int
    next_id: int
    ids: np.ndarray  # (K,)

    @staticmethod
    def empty() -> "SafeZones":
        return SafeZones(
            pos=np.zeros((0, 2), dtype=np.float64),
            vel=np.zeros((0, 2), dtype=np.float64),
            cap=np.zeros((0,), dtype=np.int32),
            occ=np.zeros((0,), dtype=np.int32),
            active=np.zeros((0,), dtype=bool),
            active_total=0,
            next_id=0,
            ids=np.zeros((0,), dtype=np.int32),
        )

    def active_count(self) -> int:
        return int(self.active_total)

    def step_move_g0(
        self,
        *,
        cfg: SafeZoneConfig,
        world_size: np.ndarray,
        boundary: str,
        rng: np.random.Generator,
        t: int,
        dt: float,
    ) -> None:
        if self.pos.shape[0] == 0:
            return

        # Turn occasionally
        if cfg.turn_every > 0 and (t % cfg.turn_every) == 0:
            angles = rng.uniform(-math.radians(cfg.turn_max_deg), math.radians(cfg.turn_max_deg), size=(self.pos.shape[0],))
            c = np.cos(angles)
            s = np.sin(angles)
            vx = self.vel[:, 0] * c - self.vel[:, 1] * s
            vy = self.vel[:, 0] * s + self.vel[:, 1] * c
            self.vel = np.stack([vx, vy], axis=1)
            self.vel = cfg.move_speed * unit(self.vel)

        self.pos = self.pos + self.vel * dt

        if boundary == "periodic":
            apply_periodic_inplace(self.pos, world_size)
        elif boundary == "reflecting":
            # Reflecting for points: fold positions; velocity flips handled by sign detection
            p = self.pos
            v = self.vel
            for axis in (0, 1):
                L = float(world_size[axis])
                x = p[:, axis]
                x_mod = np.mod(x, 2 * L)
                reflected = x_mod > L
                x_mod[reflected] = 2 * L - x_mod[reflected]
                p[:, axis] = x_mod

                v[reflected, axis] *= -1.0
            self.pos = p
            self.vel = v
        else:
            raise ValueError(f"Unknown boundary: {boundary}")

    def spawn_one(
        self,
        *,
        cfg: SafeZoneConfig,
        world_size: np.ndarray,
        boundary: str,
        rng: np.random.Generator,
        evader_count: int,
        pursuer_pos: np.ndarray,
        obstacle_pos: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        if self.active_count() >= cfg.active_max:
            return None

        cap_nom = nominal_zone_capacity(evader_count, cfg.active_max, cfg.cap_ratio)
        if cfg.cap_mode == "fixed":
            cap = cap_nom
        elif cfg.cap_mode == "poisson":
            cap = int(rng.poisson(lam=max(1.0, float(cap_nom))))
            cap = max(cfg.cap_min, cap)
        else:
            raise ValueError(f"Unknown cap_mode: {cfg.cap_mode}")

        # Sample spawn position.
        spawn_pos = None
        max_tries = 200
        for _ in range(max_tries):
            if boundary == "reflecting":
                w, h = float(world_size[0]), float(world_size[1])
                edge = int(rng.integers(0, 4))
                if edge == 0:  # left
                    x = cfg.safe_radius
                    y = float(rng.uniform(0, h))
                elif edge == 1:  # right
                    x = w - cfg.safe_radius
                    y = float(rng.uniform(0, h))
                elif edge == 2:  # bottom
                    x = float(rng.uniform(0, w))
                    y = cfg.safe_radius
                else:  # top
                    x = float(rng.uniform(0, w))
                    y = h - cfg.safe_radius
                cand = np.array([x, y], dtype=np.float64)
            else:  # periodic or default
                cand = rng.uniform(low=[0.0, 0.0], high=world_size, size=(2,)).astype(np.float64)

            ok = True
            if self.pos.shape[0] > 0:
                d2 = np.sum((self.pos[self.active] - cand[None, :]) ** 2, axis=1)
                if d2.size > 0 and float(d2.min()) < cfg.min_dist_zone * cfg.min_dist_zone:
                    ok = False

            if ok and pursuer_pos.shape[0] > 0:
                d2p = np.sum((pursuer_pos - cand[None, :]) ** 2, axis=1)
                if float(d2p.min()) < cfg.min_dist_pursuer * cfg.min_dist_pursuer:
                    ok = False

            if ok and obstacle_pos is not None and obstacle_pos.shape[0] > 0:
                d2o = np.sum((obstacle_pos - cand[None, :]) ** 2, axis=1)
                if float(d2o.min()) < cfg.min_dist_obstacle * cfg.min_dist_obstacle:
                    ok = False

            if ok:
                spawn_pos = cand
                break

        if spawn_pos is None:
            return None

        vel = cfg.move_speed * unit(rng.normal(size=(1, 2))).reshape(2)

        zid = self.next_id
        self.next_id += 1

        self.pos = np.vstack([self.pos, spawn_pos.reshape(1, 2)])
        self.vel = np.vstack([self.vel, vel.reshape(1, 2)])
        self.cap = np.append(self.cap, np.int32(cap))
        self.occ = np.append(self.occ, np.int32(0))
        self.active = np.append(self.active, True)
        self.active_total += 1
        self.ids = np.append(self.ids, np.int32(zid))

        return {"type": "zone_spawn", "zone_id": int(zid), "cap": int(cap), "x": float(spawn_pos[0]), "y": float(spawn_pos[1])}

    def deactivate_full(self, idx: int) -> dict[str, Any]:
        if self.active[idx]:
            self.active_total -= 1
        self.active[idx] = False
        return {"type": "zone_deactivate", "zone_id": int(self.ids[idx]), "x": float(self.pos[idx, 0]), "y": float(self.pos[idx, 1])}
