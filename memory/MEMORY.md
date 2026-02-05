# ChaseCrit Long-Term Memory

This file captures durable project context (decisions, conventions, and “what matters”), intended to persist across sessions.

## Project goal

Study whether evader swarms operating near a critical regime provide advantages in pursuit–evasion tasks. The immediate focus is on the evader swarm; pursuer policies start simple and can be strengthened later.

## Current baseline task (priority)

2D continuous pursuit–evasion with multiple safe zones:

- Evaders have **local observation only** and **no explicit communication**.
- Safe zones are **locally discoverable** (`r_detect`), have **unknown capacity**, and **disappear immediately** when capacity is exhausted.
- Safe zones move slowly using **G0 constant-speed random walk**.
- Safe zones **refresh randomly** with a maximum number of simultaneously active zones (`K_active_max`); with bounded domains, refresh spawns from the boundary.
- A hard constraint prevents prolonged “no safe zone” episodes (force-spawn when active count reaches 0).
- Capture is **instant** within `r_cap` (a “sustained capture for τ steps” option is planned as a configurable variant).

Canonical design spec: `doc/实验设计-容量限制缓慢移动多安全区.md`.

## Code structure (Python + uv)

- `src/chasecrit/sim.py`: simulation loop + outputs
- `src/chasecrit/policies.py`: evader / pursuer baseline policies
- `src/chasecrit/safe_zones.py`: safe zone motion + refresh logic
- `src/chasecrit/cli.py`: CLI entrypoints (`run`, `sweep-grid`, `report`)
- `src/chasecrit/report.py`: aggregation for sweeps
- `src/chasecrit/plotting.py`: plotting helpers
- `configs/base.toml`: base config template

Environment management uses `uv`:

- `uv sync`
- `uv run chasecrit ...`

## Experiments & artifacts conventions

- Raw sweep outputs go under `runs/` (not version-controlled).
- Aggregated results and figures go under `doc/results_*/` (version-controlled).
- Reader-facing results summary is written in `doc/实验结果-固定追捕者数量扫描.md`.

## Current “criticality” proxy used in reports

- `P(t)`: polarization order parameter among alive evaders.
- `χ = N_e · Var_t(P(t))`: susceptibility proxy computed as within-episode time variance of `P(t)` scaled by `N_e`.

This proxy is informative but not definitive; task dynamics (goals, capture, refresh) may distort classic phase-transition signatures.

## Known limitations (as of 2026-02-05)

- Evader neighbor interactions are O(N²); acceptable at `N_e≈128` for sweeps but may need optimization for larger N.
- Pursuer policy is currently `p0_nearest`; `p1_intercept` and cooperative allocation are planned.
- Obstacles are not implemented yet (design exists in docs, but simulation currently ignores obstacles).

