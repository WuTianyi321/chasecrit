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
- Task-internal `w_align` (merged seeds=200, speed_ratio 1.0~1.2): `doc/results_20260206_walign_task_internal_200seeds/`
- Pressure contrast `w_align` (speed_ratio 0.9/1.3 and 1.4 expanded to seeds=240): `doc/results_20260206_walign_pressure_091314_sr14_240seeds/`
- LaTeX manuscript draft:
  - source: `paper/main.tex`
  - compiled PDF: `paper/main.pdf`
  - updated to include all major result routes (task-internal `w_align`, pressure contrast, noise-route strong positive correlation, phase-vs-task contrast).
  - latest revision adds a broad appendix index (R1~R13) and fixes noise summary table readability (explicit peak locations/values; phase safe marked N/A).

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
- Task-internal `w_align` expansion:
  - `runs/sweep_20260206_151534_grid/` (speed_ratio 1.0~1.2, seeds=80)
  - `runs/sweep_20260206_154729_grid/` (supplement seeds 80..199)
  - merged: `runs/sweep_20260206_merged_200seeds_grid/` (21000 rows)
- Pressure contrast:
  - `runs/sweep_20260206_160622_grid/` (speed_ratio 0.9/1.3/1.4, seeds=120)
  - `runs/sweep_20260206_161736_grid/` (speed_ratio 1.4, extra seeds 120..239)
  - merged: `runs/sweep_20260206_merged_pressure_sr14_240_grid/` (10080 rows)

## Current hypothesis status (criticality vs performance)

- In the current baseline task, “closer to phase-identified critical noise” does **not** improve `safe_frac` in the tested settings (`w_align=0.6` and `w_align=1.0`, `speed_ratio=1.1`, `seeds=100`). Task success peaks at low noise (`angle_noise≈0..0.2`), while phase proxy χ peaks near `angle_noise≈1.8` in the phase-identification setting (`seeds=100`, `steps=1200`).
- Important nuance (B2: `w_align=1.0`): within the task noise sweep itself, `safe_mean` and `chi_mean` are strongly positively correlated across noise points (`Pearson r≈0.824`, `Spearman ρ≈0.965`). This supports an in-task positive coupling, but does not imply phase critical-point transferability.
- Current interpretation frame for adversarial tasks:
  - Separate **autonomous criticality** (weak/no external forcing) from **forced order** (strong external forcing causes predictable behavior).
  - Positive `safe`-`chi` correlation in-task can coexist with non-transferability of phase critical point.
  - Pursuer strength matters: with weak pursuer (`p0_nearest`), predictability penalties may not surface.
- Working interpretation:
  - `supports`: in-task low-noise region can simultaneously have higher `safe` and higher `chi`.
  - `does_not_support_yet`: using the phase-identified critical noise (`≈1.8`) as a direct recipe for better task performance.
- Next emphasis requested by user: prioritize the autonomy route (neighbor-following parameter scan, i.e. `w_align`-based near-critical tuning) and investigate which task scenarios make critical-like regimes beneficial.
- Detailed reader-facing write-up: `doc/实验结果-临界性验证-噪声控制参.md`.
- New task-internal `w_align` evidence (2026-02-06 evening):
  - For `speed_ratio=1.0~1.3`, `safe` and `chi` remain positively correlated (group-mean Pearson roughly `0.27~0.62`), supporting a conditional near-critical benefit.
  - For stronger pressure `speed_ratio=1.4`, this relation collapses (`corr(safe,chi)≈0.02`) and `safe` vs `xi` becomes negative.
  - Interpretation updated from “global yes/no” to “pressure-dependent regime”: near-critical benefit is plausible under moderate pressure but not robust under very strong pressure.
  - `w_align` point optimum remains broad/flat under current noise; conclusions should be stated as an interval/band, not a single exact optimum.

## Performance baseline (2026-02-06)

- Added optional Numba acceleration for dense neighbor interactions in `evader_step`.
- Added second-round runtime optimizations focused on allocation reduction and high-frequency math paths:
  - in-place periodic mapping (`apply_periodic_inplace`);
  - reusable capture / zone-entry buffers in `Simulation`;
  - O(1) safe-zone active count cache (`active_total`);
  - reduced repeated normalization overhead in `evader_step`.
- Added `bench-speed` CLI and validation tests.
- Reproducible benchmark artifact:
  - `doc/bench_20260206_numba_speed_v2.json`
  - `doc/工程性能-加速与一致性验证.md`
- Latest speedup (small-scale, current baseline):
  - `evader_step`: ~`4.08x`
  - end-to-end simulation: ~`2.80x`
- Before/after optimization comparison artifacts:
  - before: `runs/bench_before_opt.json`
  - after: `runs/bench_after_opt2.json`
- Behavior check on benchmark seeds shows no count mismatches and negligible floating differences.

## Active research plan (in-task criterion)

- Decision rule (current): define “more critical” by **task-internal observed statistics** (primary: `chi`; auxiliary: `chi_local`, `tau_P_ar1`, `xi_fluct`) within the same task setting.
- Immediate tasks:
  1. Continue pressure-resolution sweep around `speed_ratio=1.3~1.5` (finer grid, e.g. step `0.05`) to locate where near-critical coupling breaks.
  2. Upgrade pursuer strategy (predictive/cooperative) and rerun key `w_align` slices to test “forced-order penalty”.
  3. Validate robustness in `reflecting` + obstacle settings with the same task-internal criterion.
- Reporting principle: avoid mixing “external phase baseline” and “in-task criticality ranking” in one conclusion unless explicitly separated.
