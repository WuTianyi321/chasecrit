from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, load_config
from .geometry import apply_periodic_inplace, apply_reflecting, unit, wrap_delta_inplace
from .io_utils import csv_write, ensure_dir, json_dump
from .metrics import component_count, correlation_length_fluctuation, local_polarization_stats, polarization
from .policies import (
    EvaderDenseWorkspace,
    EvaderSocState,
    evader_step,
    pursuer_step_p0_nearest,
    pursuer_step_p1_intercept,
)
from .safe_zones import SafeZones


class Simulation:
    def __init__(self, cfg: ExperimentConfig, *, prefer_numba: bool | None = None):
        self.cfg = cfg
        self.prefer_numba = prefer_numba
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
        self._probe_evader: int = 0

        self._capture_delta = np.empty((Ne, max(1, Np), 2), dtype=np.float64)
        self._capture_dist2 = np.empty((Ne, max(1, Np)), dtype=np.float64)
        self._capture_tmp = np.empty((Ne, max(1, Np)), dtype=np.float64)
        self._capture_near = np.empty((Ne, max(1, Np)), dtype=bool)
        self._zone_delta = np.empty((Ne, 2), dtype=np.float64)
        self._zone_dist2 = np.empty((Ne,), dtype=np.float64)
        self._zone_tmp = np.empty((Ne,), dtype=np.float64)
        self._zone_within = np.empty((Ne,), dtype=bool)

        self.zones = SafeZones.empty()

        self.events: list[dict[str, Any]] = []
        self.timeseries: list[dict[str, Any]] = []

        # Online stats for polarization (Welford)
        self._P_n = 0
        self._P_mean = 0.0
        self._P_M2 = 0.0
        self._P_prev: float | None = None
        self._P_lag1_sum = 0.0

        # Always record final counts; optionally store full timeseries/events
        self._last_alive = int(self.alive.sum())
        self._last_safe = 0
        self._last_captured = 0
        self._steps_recorded = 0

        self._evader_dense_ws: EvaderDenseWorkspace | None = None
        if int(cfg.evaders.count) <= 512:
            self._evader_dense_ws = EvaderDenseWorkspace.create(int(cfg.evaders.count))

        self._soc_state: EvaderSocState | None = None
        self._soc_topple_sizes: list[int] = []
        if bool(cfg.evaders.soc_enabled):
            self._soc_state = EvaderSocState.create(
                n_evaders=int(cfg.evaders.count),
                init_align_share=float(np.clip(cfg.evaders.w_align, 0.0, 1.0)),
            )

    def _spawn_zones(self, t: int) -> None:
        cfgz = self.cfg.safe_zones
        boundary = self.cfg.world.boundary
        active_count = self.zones.active_count()

        # Hard constraint: never allow 0 active zones
        if active_count == 0 and cfgz.active_max > 0:
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
                active_count += 1

        if active_count >= cfgz.active_max:
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

        if self._P_prev is not None:
            self._P_lag1_sum += float(P) * float(self._P_prev)
        self._P_prev = float(P)

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
            probe_active = bool(self.alive[self._probe_evader] and (not self.safe[self._probe_evader]) and (not self.captured[self._probe_evader]))
            probe_heading_deg = float("nan")
            if probe_active:
                vx = float(self.vel_e[self._probe_evader, 0])
                vy = float(self.vel_e[self._probe_evader, 1])
                if (vx * vx + vy * vy) > 1e-12:
                    probe_heading_deg = float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)

            self.timeseries.append(
                {
                    "t": int(t),
                    "alive": self._last_alive,
                    "safe": self._last_safe,
                    "captured": self._last_captured,
                    "active_zones": int(self.zones.active_count()),
                    "P": P,
                    "probe_id": int(self._probe_evader),
                    "probe_active": int(probe_active),
                    "probe_heading_deg": probe_heading_deg,
                    "soc_topples": int(self._soc_state.last_topple_count) if self._soc_state is not None else 0,
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
            align_control_mode=cfg.evaders.align_control_mode,
            soc_enabled=cfg.evaders.soc_enabled,
            soc_state=self._soc_state,
            soc_stress_decay=cfg.evaders.soc_stress_decay,
            soc_surprise_gain=cfg.evaders.soc_surprise_gain,
            soc_threat_gain=cfg.evaders.soc_threat_gain,
            soc_threshold=cfg.evaders.soc_threshold,
            soc_release=cfg.evaders.soc_release,
            soc_neighbor_coupling=cfg.evaders.soc_neighbor_coupling,
            soc_align_drop=cfg.evaders.soc_align_drop,
            soc_align_relax=cfg.evaders.soc_align_relax,
            soc_align_min=cfg.evaders.soc_align_min,
            soc_align_max=cfg.evaders.soc_align_max,
            soc_topple_noise=cfg.evaders.soc_topple_noise,
            dense_ws=self._evader_dense_ws,
            prefer_numba=self.prefer_numba,
        )
        if self._soc_state is not None:
            self._soc_topple_sizes.append(int(self._soc_state.last_topple_count))
        self.pos_e = self.pos_e + self.vel_e * dt

        if self.cfg.world.boundary == "periodic":
            apply_periodic_inplace(self.pos_e, self.world_size)
        else:
            self.pos_e, self.vel_e = apply_reflecting(self.pos_e, self.vel_e, self.world_size)

        # Pursuers
        v_p = cfg.pursuers.speed_ratio * cfg.evaders.speed
        if cfg.pursuers.policy == "p0_nearest":
            self.vel_p = pursuer_step_p0_nearest(
                pos_p=self.pos_p,
                pos_e=self.pos_e,
                alive_mask=alive_mask,
                world_size=self.world_size,
                periodic=self.periodic,
                v_p=v_p,
            )
        elif cfg.pursuers.policy == "p1_intercept":
            self.vel_p = pursuer_step_p1_intercept(
                pos_p=self.pos_p,
                pos_e=self.pos_e,
                vel_e=self.vel_e,
                alive_mask=alive_mask,
                world_size=self.world_size,
                periodic=self.periodic,
                v_p=v_p,
            )
        else:
            raise ValueError(f"Unsupported pursuer policy: {cfg.pursuers.policy}")
        self.pos_p = self.pos_p + self.vel_p * dt
        if self.cfg.world.boundary == "periodic":
            apply_periodic_inplace(self.pos_p, self.world_size)
        else:
            self.pos_p, self.vel_p = apply_reflecting(self.pos_p, self.vel_p, self.world_size)

        # Capture (instant)
        if alive_mask.any() and self.pos_p.shape[0] > 0:
            npurs = self.pos_p.shape[0]
            ep = self._capture_delta[:, :npurs, :]
            np.subtract(self.pos_p[None, :, :], self.pos_e[:, None, :], out=ep)
            if self.periodic:
                wrap_delta_inplace(ep, self.world_size)

            dist2 = self._capture_dist2[:, :npurs]
            np.multiply(ep[:, :, 0], ep[:, :, 0], out=dist2)
            tmp = self._capture_tmp[:, :npurs]
            np.multiply(ep[:, :, 1], ep[:, :, 1], out=tmp)
            np.add(dist2, tmp, out=dist2)

            near = self._capture_near[:, :npurs]
            np.less_equal(dist2, cfg.pursuers.capture_radius * cfg.pursuers.capture_radius, out=near)
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
            for zone_k in active_idx.tolist():
                if not self.zones.active[zone_k]:
                    continue

                ez = self._zone_delta
                np.subtract(self.zones.pos[zone_k], self.pos_e, out=ez)
                if self.periodic:
                    wrap_delta_inplace(ez, self.world_size)

                dist2 = self._zone_dist2
                np.multiply(ez[:, 0], ez[:, 0], out=dist2)
                np.multiply(ez[:, 1], ez[:, 1], out=self._zone_tmp)
                np.add(dist2, self._zone_tmp, out=dist2)
                np.less_equal(dist2, cfg.safe_zones.safe_radius * cfg.safe_zones.safe_radius, out=self._zone_within)

                entrants = np.flatnonzero(self._zone_within & alive_mask)
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
        rho1_P = 0.0
        tau_P_ar1 = 0.0
        if self._P_n >= 2 and P_var > 1e-12:
            lag1_mean = float(self._P_lag1_sum / (self._P_n - 1))
            cov1 = lag1_mean - P_mean * P_mean
            rho1_P = float(cov1 / P_var)
            rho1_P = float(np.clip(rho1_P, -0.99, 0.99))
            tau_P_ar1 = float((1.0 + rho1_P) / (1.0 - rho1_P))
        xi = correlation_length_fluctuation(
            pos=self.pos_e,
            vel=self.vel_e,
            alive_mask=alive_mask,
            world_size=self.world_size,
            periodic=self.periodic,
        )
        comps = component_count(
            pos=self.pos_e,
            alive_mask=alive_mask,
            world_size=self.world_size,
            periodic=self.periodic,
            radius=self.cfg.evaders.neighbor_radius,
        )
        P_local_mean, P_local_var = local_polarization_stats(
            pos=self.pos_e,
            vel=self.vel_e,
            alive_mask=alive_mask,
            world_size=self.world_size,
            periodic=self.periodic,
            neighbor_radius=self.cfg.evaders.neighbor_radius,
        )
        topple_arr = np.asarray(self._soc_topple_sizes, dtype=np.float64)
        topple_pos = topple_arr[topple_arr > 0.0]
        branch_ratio = 0.0
        if topple_pos.size >= 2:
            prev = topple_pos[:-1]
            nxt = topple_pos[1:]
            valid = prev > 0.0
            if np.any(valid):
                branch_ratio = float(np.mean(nxt[valid] / prev[valid]))
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
            "rho1_P": float(rho1_P),
            "tau_P_ar1": float(tau_P_ar1),
            "xi_fluct": float(xi),
            "components": int(comps),
            "P_local_mean": float(P_local_mean),
            "P_local_var": float(P_local_var),
            "chi_local": float(Ne0 * P_local_var),
            "soc_topple_steps": int(np.count_nonzero(topple_arr > 0.0)),
            "soc_topple_events_total": int(np.sum(topple_arr)),
            "soc_topple_size_mean": float(np.mean(topple_pos)) if topple_pos.size else 0.0,
            "soc_topple_size_var": float(np.var(topple_pos)) if topple_pos.size else 0.0,
            "soc_branch_ratio": float(branch_ratio),
        }

    def save(self, run_dir: Path) -> None:
        ensure_dir(run_dir)
        json_dump(run_dir / "config.json", self.cfg.to_dict())
        json_dump(run_dir / "summary.json", self.summary())

        if self.cfg.run.save_timeseries:
            csv_write(run_dir / "timeseries.csv", self.timeseries, fieldnames=list(self.timeseries[0].keys()) if self.timeseries else ["t"])

        if self.cfg.run.save_events:
            json_dump(run_dir / "events.json", self.events)


def run_summary(cfg: ExperimentConfig, *, prefer_numba: bool | None = None) -> dict[str, Any]:
    sim = Simulation(cfg, prefer_numba=prefer_numba)
    sim.run()
    return sim.summary()


def run_once(
    cfg: ExperimentConfig,
    *,
    run_name: str | None = None,
    save_run: bool = True,
    prefer_numba: bool | None = None,
) -> tuple[dict[str, Any], Path | None]:
    sim = Simulation(cfg, prefer_numba=prefer_numba)
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
