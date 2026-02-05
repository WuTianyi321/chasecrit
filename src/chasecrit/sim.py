from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, load_config
from .geometry import apply_periodic, apply_reflecting, unit, wrap_delta
from .io_utils import csv_write, ensure_dir, json_dump
from .metrics import polarization
from .policies import evader_step, pursuer_step_p0_nearest
from .safe_zones import SafeZones


class Simulation:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.run.seed)

        w = cfg.world.width
        h = cfg.world.height
        self.world_size = np.array([w, h], dtype=np.float64)
        self.periodic = cfg.world.boundary == "periodic"

        Ne = cfg.evaders.count
        self.pos_e = self.rng.uniform(low=[0.0, 0.0], high=self.world_size, size=(Ne, 2)).astype(np.float64)
        self.vel_e = cfg.evaders.speed * unit(self.rng.normal(size=(Ne, 2))).astype(np.float64)
        self.alive = np.ones((Ne,), dtype=bool)
        self.safe = np.zeros((Ne,), dtype=bool)
        self.captured = np.zeros((Ne,), dtype=bool)

        Np = cfg.pursuers.count
        self.pos_p = self.rng.uniform(low=[0.0, 0.0], high=self.world_size, size=(Np, 2)).astype(np.float64)
        self.vel_p = np.zeros((Np, 2), dtype=np.float64)

        self.zones = SafeZones.empty()

        self.events: list[dict[str, Any]] = []
        self.timeseries: list[dict[str, Any]] = []

        # Online stats for polarization (Welford)
        self._P_n = 0
        self._P_mean = 0.0
        self._P_M2 = 0.0

        # Always record final counts; optionally store full timeseries/events
        self._last_alive = int(self.alive.sum())
        self._last_safe = 0
        self._last_captured = 0
        self._steps_recorded = 0

    def _spawn_zones(self, t: int) -> None:
        cfgz = self.cfg.safe_zones
        boundary = self.cfg.world.boundary

        # Hard constraint: never allow 0 active zones
        if self.zones.active_count() == 0 and cfgz.active_max > 0:
            ev = self.zones.spawn_one(
                cfg=cfgz,
                world_size=self.world_size,
                boundary=boundary,
                rng=self.rng,
                evader_count=self.cfg.evaders.count,
                pursuer_pos=self.pos_p,
            )
            if ev is not None:
                ev["t"] = t
                if self.cfg.run.save_events:
                    self.events.append(ev)

        if self.zones.active_count() >= cfgz.active_max:
            return

        if self.rng.random() < cfgz.spawn_prob:
            ev = self.zones.spawn_one(
                cfg=cfgz,
                world_size=self.world_size,
                boundary=boundary,
                rng=self.rng,
                evader_count=self.cfg.evaders.count,
                pursuer_pos=self.pos_p,
            )
            if ev is not None:
                ev["t"] = t
                if self.cfg.run.save_events:
                    self.events.append(ev)

    def _update_timeseries(self, t: int) -> None:
        alive_mask = self.alive & (~self.safe) & (~self.captured)
        P = polarization(self.vel_e, alive_mask)

        # Online stats update
        self._P_n += 1
        delta = P - self._P_mean
        self._P_mean += delta / self._P_n
        self._P_M2 += delta * (P - self._P_mean)

        self._last_alive = int(alive_mask.sum())
        self._last_safe = int(self.safe.sum())
        self._last_captured = int(self.captured.sum())
        self._steps_recorded = int(t) + 1

        if self.cfg.run.save_timeseries:
            self.timeseries.append(
                {
                    "t": int(t),
                    "alive": self._last_alive,
                    "safe": self._last_safe,
                    "captured": self._last_captured,
                    "active_zones": int(self.zones.active_count()),
                    "P": P,
                }
            )

    def step(self, t: int) -> None:
        cfg = self.cfg
        dt = cfg.run.dt

        # Move safe zones + spawn
        self.zones.step_move_g0(cfg=cfg.safe_zones, world_size=self.world_size, boundary=cfg.world.boundary, rng=self.rng, t=t, dt=dt)
        self._spawn_zones(t)

        alive_mask = self.alive & (~self.safe) & (~self.captured)

        # Evaders
        self.vel_e = evader_step(
            pos_e=self.pos_e,
            vel_e=self.vel_e,
            alive_mask=alive_mask,
            pos_p=self.pos_p,
            pos_z=self.zones.pos,
            zone_active=self.zones.active,
            world_size=self.world_size,
            periodic=self.periodic,
            rng=self.rng,
            v_e=cfg.evaders.speed,
            inertia=cfg.evaders.inertia,
            r_nbr=cfg.evaders.neighbor_radius,
            r_pred=cfg.evaders.pred_radius,
            r_detect=cfg.evaders.detect_radius,
            r_sep=cfg.evaders.sep_radius,
            w_align=cfg.evaders.w_align,
            w_goal=cfg.evaders.w_goal,
            w_avoid=cfg.evaders.w_avoid,
            w_explore=cfg.evaders.w_explore,
            sep_strength=cfg.evaders.sep_strength,
            angle_noise=cfg.evaders.angle_noise,
        )
        self.pos_e = self.pos_e + self.vel_e * dt

        if self.cfg.world.boundary == "periodic":
            self.pos_e = apply_periodic(self.pos_e, self.world_size)
        else:
            self.pos_e, self.vel_e = apply_reflecting(self.pos_e, self.vel_e, self.world_size)

        # Pursuers (P0 for now)
        v_p = cfg.pursuers.speed_ratio * cfg.evaders.speed
        self.vel_p = pursuer_step_p0_nearest(
            pos_p=self.pos_p,
            pos_e=self.pos_e,
            alive_mask=alive_mask,
            world_size=self.world_size,
            periodic=self.periodic,
            v_p=v_p,
        )
        self.pos_p = self.pos_p + self.vel_p * dt
        if self.cfg.world.boundary == "periodic":
            self.pos_p = apply_periodic(self.pos_p, self.world_size)
        else:
            self.pos_p, self.vel_p = apply_reflecting(self.pos_p, self.vel_p, self.world_size)

        # Capture (instant)
        if alive_mask.any() and self.pos_p.shape[0] > 0:
            ep = self.pos_p[None, :, :] - self.pos_e[:, None, :]
            if self.periodic:
                ep = wrap_delta(ep, self.world_size)
            dist2 = np.sum(ep**2, axis=2)
            near = dist2 <= (cfg.pursuers.capture_radius * cfg.pursuers.capture_radius)
            to_cap = alive_mask & near.any(axis=1)
            if to_cap.any():
                idxs = np.flatnonzero(to_cap)
                self.captured[idxs] = True
                if cfg.run.save_events:
                    for i in idxs.tolist():
                        self.events.append(
                            {"t": int(t), "type": "capture", "evader": int(i), "x": float(self.pos_e[i, 0]), "y": float(self.pos_e[i, 1])}
                        )

        # Enter safe zone (process by zone, random tie-break)
        alive_mask = self.alive & (~self.safe) & (~self.captured)
        if alive_mask.any() and self.zones.active_count() > 0:
            active_idx = np.flatnonzero(self.zones.active)
            zpos = self.zones.pos[active_idx]
            ez = zpos[None, :, :] - self.pos_e[:, None, :]
            if self.periodic:
                ez = wrap_delta(ez, self.world_size)
            dist2 = np.sum(ez**2, axis=2)
            within = dist2 <= (cfg.safe_zones.safe_radius * cfg.safe_zones.safe_radius)
            for local_k, zone_k in enumerate(active_idx.tolist()):
                entrants = np.flatnonzero(within[:, local_k] & alive_mask)
                if entrants.size == 0:
                    continue
                self.rng.shuffle(entrants)
                for i in entrants.tolist():
                    if not self.zones.active[zone_k]:
                        break
                    if int(self.zones.occ[zone_k]) < int(self.zones.cap[zone_k]):
                        self.zones.occ[zone_k] += 1
                        self.safe[i] = True
                        if cfg.run.save_events:
                            self.events.append(
                                {"t": int(t), "type": "enter_safe", "evader": int(i), "zone_id": int(self.zones.ids[zone_k])}
                            )
                        if int(self.zones.occ[zone_k]) >= int(self.zones.cap[zone_k]):
                            ev = self.zones.deactivate_full(zone_k)
                            ev["t"] = int(t)
                            if cfg.run.save_events:
                                self.events.append(ev)
                    else:
                        if self.zones.active[zone_k]:
                            ev = self.zones.deactivate_full(zone_k)
                            ev["t"] = int(t)
                            if cfg.run.save_events:
                                self.events.append(ev)
                        break

        # Enforce "never stays without safe zones": if all zones disappeared this step,
        # refresh immediately (and still allow probabilistic spawn).
        self._spawn_zones(t)

        self._update_timeseries(t)

    def run(self) -> dict[str, Any]:
        T = self.cfg.run.steps
        for t in range(T):
            alive_mask = self.alive & (~self.safe) & (~self.captured)
            if alive_mask.sum() == 0:
                self._update_timeseries(t)
                break
            self.step(t)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        alive_mask = self.alive & (~self.safe) & (~self.captured)
        P_mean = float(self._P_mean) if self._P_n else 0.0
        P_var = float(self._P_M2 / self._P_n) if self._P_n else 0.0
        Ne0 = int(self.cfg.evaders.count)
        return {
            "alive": int(alive_mask.sum()),
            "safe": int(self.safe.sum()),
            "captured": int(self.captured.sum()),
            "steps_recorded": int(self._steps_recorded),
            "safe_frac": float(self.safe.sum() / Ne0) if Ne0 else 0.0,
            "captured_frac": float(self.captured.sum() / Ne0) if Ne0 else 0.0,
            "P_mean": P_mean,
            "P_var": P_var,
            "chi": float(Ne0 * P_var),
        }

    def save(self, run_dir: Path) -> None:
        ensure_dir(run_dir)
        json_dump(run_dir / "config.json", self.cfg.to_dict())
        json_dump(run_dir / "summary.json", self.summary())

        if self.cfg.run.save_timeseries:
            csv_write(run_dir / "timeseries.csv", self.timeseries, fieldnames=list(self.timeseries[0].keys()) if self.timeseries else ["t"])

        if self.cfg.run.save_events:
            json_dump(run_dir / "events.json", self.events)


def run_summary(cfg: ExperimentConfig) -> dict[str, Any]:
    sim = Simulation(cfg)
    sim.run()
    return sim.summary()


def run_once(
    cfg: ExperimentConfig,
    *,
    run_name: str | None = None,
    save_run: bool = True,
) -> tuple[dict[str, Any], Path | None]:
    sim = Simulation(cfg)
    sim.run()

    summary = sim.summary()
    if not save_run:
        return summary, None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if run_name is None:
        run_name = f"run_{ts}_seed{cfg.run.seed}"
    run_dir = Path(cfg.run.out_dir) / run_name
    sim.save(run_dir)
    return summary, run_dir
