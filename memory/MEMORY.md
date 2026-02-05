# ChaseCrit Long-Term Memory

This is the **only reliable long-term memory** for the project. Future sessions may not retain any other context.

Rule: if something might be needed later (progress, commands, results paths, decisions, pitfalls), it must be written here or in `memory/YYYY-MM-DD.md`.

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
- Long sweeps must write progress to files inside the sweep dir (`progress.jsonl`, `progress.txt`) so ETA is visible while running.

## Repository / GitHub

- Branch: `main`
- Remote: `origin` (GitHub)
- Recent notable commits:
  - `e8df685` add sim engine + CLI + uv
  - `13ee1c4` add experiment docs + initial fixed-`N_p` results
  - `9ddcf34` add AGENTS + memory system
  - `5ed3dec` speed up sweeps + progress logging

## Current “criticality” proxy used in reports

- `P(t)`: polarization order parameter among alive evaders.
- `χ = N_e · Var_t(P(t))`: susceptibility proxy computed as within-episode time variance of `P(t)` scaled by `N_e`.

This proxy is informative but not definitive; task dynamics (goals, capture, refresh) may distort classic phase-transition signatures.

## Known limitations (as of 2026-02-05)

- Evader neighbor interactions are O(N²); acceptable at `N_e≈128` for sweeps but may need optimization for larger N.
- As of `5ed3dec`, neighbor interactions are optimized: compute on alive subset and use vectorized pairwise when `alive<=256` (fast); fall back to cell-binning loops for larger swarms.
- Pursuer policy is currently `p0_nearest`; `p1_intercept` and cooperative allocation are planned.
- Obstacles are not implemented yet (design exists in docs, but simulation currently ignores obstacles).

## Results snapshots (paths)

Reader-facing summary:
- `doc/实验结果-固定追捕者数量扫描.md`

Aggregated report + figures:
- Small grid: `doc/results_20260205_fixedNp_scan/`
- Expanded grid (includes χ plots): `doc/results_20260205_fixedNp_scan_expanded/`
- Phase identification (noise control): `doc/results_20260205_phase_noise_wider/`
- Task validation (noise sweep, w_align=1.0): `doc/results_20260205_task_noise_walign1/`

Raw sweeps (not version-controlled):
- `runs/sweep_20260205_135921_grid/` (330 runs)
- `runs/sweep_20260205_154643_grid/` (1575 runs)
- Phase noise sweeps:
  - `runs/sweep_20260205_171832_phase_noise/` (noise 0..3 step 0.2, seeds=20)
- Task noise sweep:
  - `runs/sweep_20260205_172202_noise/` (speed_ratio 1.0/1.1/1.2 × noise 0..3 step 0.2, seeds=15, w_align=1.0)

## Current hypothesis status (criticality vs performance)

- In the current baseline task, “closer to phase-identified critical noise” does **not** improve `safe_frac` in the tested setting (`w_align=1.0`). Task success peaks at low noise (`angle_noise≈0..0.2`), while phase proxy χ peaks near `angle_noise≈1.8` in the phase-identification setting.
- Detailed reader-facing write-up: `doc/实验结果-临界性验证-噪声控制参.md`.
