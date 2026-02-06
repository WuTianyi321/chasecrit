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
  - `progress.txt` includes `eta_at=<timestamp>` for quick wall-clock estimates.

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
- `χ = N_e · Var_t(P(t))`: global susceptibility proxy computed as within-episode time variance of `P(t)` scaled by `N_e`.
- `χ_local = N_e · Var(P_local)`: local susceptibility proxy based on per-evader local polarization variance at the final state.
- `ξ` (`xi_fluct`): spatial correlation-length proxy from velocity-direction fluctuation correlations (final state).
- `components`: connected-component count in the evader neighbor graph (final state).
- `rho1_P`, `tau_P_ar1`: lag-1 autocorrelation and AR(1)-based correlation-time proxy from the within-episode `P(t)` series.

This proxy is informative but not definitive; task dynamics (goals, capture, refresh) may distort classic phase-transition signatures.

## Known limitations (as of 2026-02-05)

- Evader neighbor interactions are O(N²) in the dense branch; acceptable at `N_e≈128` but may need more work for larger swarms.
- Performance work so far:
  - Dense branch uses scratch buffers (`EvaderDenseWorkspace`) to reduce per-step allocations while preserving behavior.
  - For large swarms (`alive>256`), a cell-binning fallback avoids O(N²) distance matrices.
- Pursuer policy is currently `p0_nearest`; `p1_intercept` and cooperative allocation are planned.
- Obstacles are not implemented yet (design exists in docs, but simulation currently ignores obstacles).

## Results snapshots (paths)

Reader-facing summary:
- `doc/实验结果-固定追捕者数量扫描.md`
- `doc/实验结果-临界性验证-噪声控制参.md`

Aggregated report + figures:
- Small grid: `doc/results_20260205_fixedNp_scan/`
- Expanded grid (includes χ plots): `doc/results_20260205_fixedNp_scan_expanded/`
- Fixed-`N_p` scan (seeds=50, added χ_local/ξ): `doc/results_20260205_fixedNp_scan_50seeds_v2/`
- Phase identification (noise control): `doc/results_20260205_phase_noise_wider/`
- Task validation (noise sweep, w_align=1.0): `doc/results_20260205_task_noise_walign1/`
- Fixed-`N_p` scan (seeds=100, speed_ratio=1.1/1.2, added τ): `doc/results_20260206_fixedNp_sr11_sr12_100seeds/`
- Phase identification (noise, seeds=100, steps=1200): `doc/results_20260206_phase_noise_100seeds_steps1200/`
- Task noise (w_align=0.6, speed_ratio=1.1, seeds=100): `doc/results_20260206_task_noise_w06_sr11_100seeds/`
- Task noise (w_align=1.0, speed_ratio=1.1, seeds=100): `doc/results_20260206_task_noise_w10_sr11_100seeds/`

Raw sweeps (not version-controlled):
- `runs/sweep_20260205_135921_grid/` (330 runs)
- `runs/sweep_20260205_154643_grid/` (1575 runs)
- `runs/sweep_20260205_180558_grid/` (5250 runs; seeds=50 grid)
- `runs/sweep_20260206_132212_grid/` (4200 runs; speed_ratio 1.1/1.2 × w_align × seeds=100)
- Phase noise sweeps:
  - `runs/sweep_20260205_171832_phase_noise/` (noise 0..3 step 0.2, seeds=20)
  - `runs/sweep_20260206_133544_phase_noise/` (noise 0..3 step 0.2, seeds=100, steps=1200)
- Task noise sweep:
  - `runs/sweep_20260205_172202_noise/` (speed_ratio 1.0/1.1/1.2 × noise 0..3 step 0.2, seeds=15, w_align=1.0)
  - `runs/sweep_20260206_135213_noise/` (speed_ratio=1.1 × noise 0..3 step 0.2, seeds=100, w_align=0.6)
  - `runs/sweep_20260206_135744_noise/` (speed_ratio=1.1 × noise 0..3 step 0.2, seeds=100, w_align=1.0)

## Current hypothesis status (criticality vs performance)

- In the current baseline task, “closer to phase-identified critical noise” does **not** improve `safe_frac` in the tested settings (`w_align=0.6` and `w_align=1.0`, `speed_ratio=1.1`, `seeds=100`). Task success peaks at low noise (`angle_noise≈0..0.2`), while phase proxy χ peaks near `angle_noise≈1.8` in the phase-identification setting (`seeds=100`, `steps=1200`).
- Detailed reader-facing write-up: `doc/实验结果-临界性验证-噪声控制参.md`.
