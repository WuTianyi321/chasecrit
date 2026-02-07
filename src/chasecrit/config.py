from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import tomllib
from typing import Any, Literal


BoundaryMode = Literal["periodic", "reflecting"]
PursuerPolicy = Literal["p0_nearest", "p1_intercept"]
CapMode = Literal["fixed", "poisson"]
AlignControlMode = Literal["legacy", "share"]


@dataclass(frozen=True)
class WorldConfig:
    width: float = 200.0
    height: float = 200.0
    boundary: BoundaryMode = "periodic"


@dataclass(frozen=True)
class EvaderConfig:
    count: int = 128
    speed: float = 1.0
    inertia: float = 0.35

    neighbor_radius: float = 6.0
    pred_radius: float = 10.0
    detect_radius: float = 4.0

    sep_radius: float = 1.25
    sep_strength: float = 1.0

    w_align: float = 0.6
    w_goal: float = 1.2
    w_avoid: float = 1.5
    w_explore: float = 0.4
    align_control_mode: AlignControlMode = "legacy"

    angle_noise: float = 0.12  # radians

    # Optional SOC controller (disabled by default for backward compatibility).
    soc_enabled: bool = False
    soc_stress_decay: float = 0.05
    soc_surprise_gain: float = 0.4
    soc_threat_gain: float = 0.3
    soc_threshold: float = 0.25
    soc_release: float = 0.2
    soc_neighbor_coupling: float = 0.15
    soc_align_drop: float = 0.25
    soc_align_relax: float = 0.02
    soc_align_min: float = 0.05
    soc_align_max: float = 0.95
    soc_topple_noise: float = 0.2  # radians


@dataclass(frozen=True)
class PursuerConfig:
    count: int = 2
    speed_ratio: float = 1.1  # v_p = speed_ratio * v_e
    policy: PursuerPolicy = "p0_nearest"
    capture_radius: float = 1.0


@dataclass(frozen=True)
class SafeZoneConfig:
    active_max: int = 4
    safe_radius: float = 1.0
    move_speed: float = 0.1  # absolute, in same units as v_e
    turn_every: int = 50
    turn_max_deg: float = 10.0

    spawn_prob: float = 0.02
    cap_ratio: float = 1.0  # nominal: active_max * E[cap] ~= cap_ratio * N_e
    cap_mode: CapMode = "fixed"
    cap_min: int = 1

    min_dist_obstacle: float = 2.0
    min_dist_pursuer: float = 8.0
    min_dist_zone: float = 6.0


@dataclass(frozen=True)
class RunConfig:
    steps: int = 2000
    dt: float = 1.0
    seed: int = 0
    out_dir: str = "runs"
    save_timeseries: bool = True
    save_events: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    evaders: EvaderConfig = field(default_factory=EvaderConfig)
    pursuers: PursuerConfig = field(default_factory=PursuerConfig)
    safe_zones: SafeZoneConfig = field(default_factory=SafeZoneConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    raw: dict[str, Any]
    if p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() in {".toml", ".tml"}:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config file type: {p.suffix}")

    merged = _merge_dict(ExperimentConfig().to_dict(), raw)

    return ExperimentConfig(
        world=WorldConfig(**merged["world"]),
        evaders=EvaderConfig(**merged["evaders"]),
        pursuers=PursuerConfig(**merged["pursuers"]),
        safe_zones=SafeZoneConfig(**merged["safe_zones"]),
        run=RunConfig(**merged["run"]),
    )


def nominal_zone_capacity(evader_count: int, active_max: int, cap_ratio: float) -> int:
    if active_max <= 0:
        return 0
    return max(1, int(math.floor((cap_ratio * evader_count) / active_max)))
