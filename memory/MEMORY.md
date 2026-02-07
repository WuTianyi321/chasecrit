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
- Pursuer policies now include `p0_nearest` and `p1_intercept`; cooperative allocation remains planned.
- Obstacles are not implemented yet (design exists in docs, but simulation currently ignores obstacles).

## Results snapshots (paths)

Reader-facing summary:
- `doc/实验结果-固定追捕者数量扫描.md`
- `doc/实验结果-临界性验证-噪声控制参.md`
- `doc/实验结果总览-指标含义与关系解释.md`

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
  - source: `paper/main.tex` + section files under `paper/sections/`
  - compiled PDF: `paper/main.pdf`
  - manuscript is organized as subdocuments (`00_abstract` ... `06_reproducibility`) for maintainable editing.
  - results section now covers all completed batches using unified IDs:
    - quantitative: `E01..E12`
    - pilot/process validation: `P01..P06`
  - corrected noise summary table semantics:
    - `task_noise_w10_sr11_100seeds`: `argmax safe` at noise `0.2`, `argmax chi` at noise `0.0`
    - `task_noise_w06_sr11_100seeds`: both peaks at noise `0.0`
    - `phase_noise_100seeds_steps1200`: `chi` peak at noise `1.8`; safe marked not-applicable in manuscript summary.
  - discussion now explicitly separates:
    - task-internal “higher chi => more near-critical” interpretation; and
    - non-transferability of external phase critical point to task optimum.

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
- E09 high-chi spread diagnosis:
  - The apparent large `safe` spread around `chi≈4.5` in pooled E09 scatter is mainly a mixed-condition effect (different `speed_ratio` layers overlaid), not pure sampling noise.
  - In `chi∈[4.2,4.8]`, variance decomposition on group means shows about `96.7%` of `safe` variance comes from between-`speed_ratio` differences.
  - Bootstrap on raw runs indicates:
    - `sr=0.9`: `corr(safe,chi)` positive (95% CI above 0);
    - `sr=1.3`: `corr(safe,chi)` positive (95% CI above 0);
    - `sr=1.4`: correlation near 0 (95% CI crosses 0), consistent with high-pressure decoupling.
  - Reader-facing analysis note: `doc/实验结果-E09-chi波动与样本量分析.md`.
 - Raw-point (non-aggregated) view:
  - Using per-run `results.csv` points, `corr(chi, safe_frac)` is near zero in E09 (overall about `-0.016`; per-speed ratios also weak), so the relation is not visible at raw trajectory level.
  - Interpretation should separate:
    - raw-run stochastic cloud; and
    - grouped multi-seed parameter-level trend.
- Consolidated interpretation (from synthesis report):
  - “Indicators unrelated to performance” is an overstatement; the relation is strongly condition-dependent.
  - Main reasons for weak/unstable apparent coupling:
    1. mixed pressure layers in pooled views;
    2. metric-task mismatch (state descriptors vs terminal objective);
    3. non-monotonic or plateau-shaped parameter-response curves;
    4. strong run-level stochasticity in finite-horizon adversarial episodes.

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
- Manuscript method-detail requirement (implemented): keep movement/capture/zone dynamics equations explicit enough for direct reimplementation; avoid high-level prose-only method summaries.
- Single-parameter route (implemented, for next experiments):
  - new evader control mode: `align_control_mode = "share"`.
  - In this mode, `w_align` is interpreted as alignment-share `λ∈[0,1]` in
    `d = λ d_align + (1-λ) d_non_align`.
  - Recommended protocol for clarity: set `angle_noise = 0`, keep non-align weights fixed, then scan `w_align`.
  - Config scaffold: `configs/walign_share_noise0.toml`.
  - Planning note: `doc/实验路线梳理与下一步协议.md`.
- First share-mode run completed:
  - sweep: `runs/sweep_20260206_210225_grid/` (sr `0.9/1.1/1.3/1.4`, `w_align` step `0.05`, seeds `80`, steps `600`).
  - report: `doc/results_20260206_walign_share_noise0_sr09111314_80seeds/`.
  - finding: best survival at low-mid `w_align` (`~0.15-0.35`), while `w_align=1.0` is strongly suboptimal.
  - `corr(safe, chi)` in this setting is negative across all pressure layers, indicating high alignment share does not imply better task performance.
- New metric direction (from literature):
  - Source reviewed: `doc/1-s2.0-S0378437125007290-main.pdf` (Physica A, predictive entropy variance).
  - Candidate primary indicator for this project: sequence-model predictive entropy variance `Var(H_pred)` computed from local/single-agent trajectories.
  - Rationale: in complex adversarial tasks, predictability fluctuation may be more informative than pure global-order fluctuations.
  - Implementation roadmap and protocol note: `doc/临界性指标扩展-预测熵方差与任务适配.md`.
  - MVP implementation status:
    - command: `chasecrit report-pred-entropy`
    - module: `src/chasecrit/predictive_entropy.py`
    - requires per-run timeseries (`--save-runs`, no `--no-save-timeseries`)
    - uses probe heading sequence + circular EMA + discretization + N-gram predictor
    - caveat: `sweep-grid` parallel path (`jobs>1`) currently does not preserve per-run `run_dir`; predictive-entropy analysis therefore requires sweeps executed with `jobs=1` when using `--save-runs`.
  - pilot evidence (`share` mode, `noise=0`, `sr=1.1`):
    - seeds=12: `corr(safe,chi)≈-0.374`, `corr(safe,Var(H_pred))≈+0.283`
    - seeds=20: `corr(safe,chi)≈-0.538`, `corr(safe,Var(H_pred))≈+0.224`
    - suggests `Var(H_pred)` may track task performance direction better than `chi` in this setting.
  - scale-up evidence (`share`, `noise=0`, `sr=1.1/1.3`, seeds=80):
    - sweep: `runs/sweep_20260206_214125_grid/` (`3360` runs).
    - predictor-valid runs: `3320`.
    - `sr=1.1`: `corr(safe,Var(H_pred))` weak negative; `sr=1.3`: moderate positive.
    - pooled correlation remains weak positive, suggesting regime dependence rather than universal monotonic coupling.

## SOC controller status (2026-02-07)

- SOC controller has been integrated as an opt-in path with backward compatibility:
  - config fields in `EvaderConfig` (`soc_enabled` and SOC gains/threshold/relaxation bounds),
  - state container `EvaderSocState` in `src/chasecrit/policies.py`,
  - simulation output metrics:
    - `soc_topple_steps`,
    - `soc_topple_events_total`,
    - `soc_topple_size_mean`,
    - `soc_topple_size_var`,
    - `soc_branch_ratio`.
  - semantic note:
    - with SOC enabled, `w_align` is the set-point for dynamic per-agent `lambda_i(t)` (toppling drops it, relaxation pulls it back), not a fixed weight.
- Compatibility and behavior tests:
  - new file `tests/test_soc_mode.py`,
  - full test status after integration: `uv run pytest -q` -> `9 passed, 3 skipped`,
  - type check: `uv run pyright` -> `0 errors`.
- SOC smoke experiment artifacts:
  - baseline: `runs/sweep_20260207_103331_grid` -> `doc/results_20260207_soc_smoke_baseline/`
  - SOC-MVP: `runs/sweep_20260207_103347_grid` -> `doc/results_20260207_soc_smoke_mvp/`
  - SOC-Soft: `runs/sweep_20260207_103434_grid` -> `doc/results_20260207_soc_smoke_soft/`
  - settings: `speed_ratio=1.1`, `w_align in {0.4,0.6,0.8}`, `steps=400`, `seeds=20`.
- Current conclusion from smoke:
  - SOC dynamics signals are present (`soc_topple_steps` high, `soc_branch_ratio` around `1.11`).
  - These two initial SOC parameterizations do **not** yet improve `safe_frac` over baseline.
  - Next work should tune SOC criticality level (threshold/coupling/drop/drive) and test across pressure layers rather than using a single fixed SOC parameter set.
- Reader-facing summary for this stage:
  - `doc/实验结果-SOC算法接入与初步冒烟实验-20260207.md`.
- SOC conceptual update:
  - new design baseline is SOC-v2 without regression to fixed `w_align` in SOC mode;
  - `w_align` should not be treated as a steady set-point for SOC claims;
  - preferred formulation maps dynamic alignment directly from stress (`lambda_i=f(s_i)`), with predictive-entropy fluctuation as local drive.
  - design note: `doc/SOC-v2方案-基于预测熵波动的自组织临界控制.md`.

## Pursuer policy status

- `p1_intercept` is now implemented and selectable via `pursuers.policy`.
  - default remains `p0_nearest` for backward compatibility.
  - implementation paths:
    - `src/chasecrit/policies.py` (`pursuer_step_p1_intercept`)
    - `src/chasecrit/sim.py` (policy dispatch).
- coverage:
  - `tests/test_pursuer_policies.py` validates intercept leading behavior and nearest fallback.

## Latest alignment-share re-run with predictive pursuer (2026-02-07)

- Experiment:
  - config: `configs/walign_share_noise0_intercept.toml`
  - sweep: `runs/sweep_20260207_110454_grid`
  - settings: `speed_ratio={0.9,1.1,1.3,1.4}`, `w_align=0..1 step 0.05`, seeds `80`, steps `600`.
  - report: `doc/results_20260207_walign_share_noise0_p1intercept_sr09111314_80seeds/`
  - summary note: `doc/实验结果-对齐比例-预测追捕复现实验-20260207.md`.
- Main comparison to prior `p0_nearest` share/noise0 run:
  - best `w_align` moves lower (`0.10~0.25`);
  - best `safe_frac` increases in all tested pressure layers;
  - `corr(safe,chi)` is more negative across all layers;
  - `w_align=1.0` remains strongly suboptimal.

## Expanded policy-comparison scans (2026-02-07, later)

- Added larger follow-up scans and explicit comparison plots.
- Fine low-`w` comparison (`w=0..0.30`, step `0.02`, seeds `120`):
  - `p1`: `runs/sweep_20260207_150007_grid` -> `doc/results_20260207_walign_share_noise0_p1intercept_sr09111314_120seeds_fine03/`
  - `p0`: `runs/sweep_20260207_151352_grid` -> `doc/results_20260207_walign_share_noise0_p0nearest_sr09111314_120seeds_fine03/`
- `p1` boundary-resolution scan (`w=0..0.60`, step `0.02`, seeds `60`):
  - `runs/sweep_20260207_152551_grid` -> `doc/results_20260207_walign_share_noise0_p1intercept_sr09111314_60seeds_w06/`
- Cross-policy compare artifacts:
  - `doc/results_20260207_walign_share_fine03_policy_compare/policy_compare_summary.csv`
  - `doc/results_20260207_walign_share_fine03_policy_compare/figs/safe_vs_w_align_policy_compare.png`
  - `doc/results_20260207_walign_share_fine03_policy_compare/figs/delta_safe_vs_w_align_policy.png`
  - `doc/results_20260207_walign_share_fine03_policy_compare/figs/chi_vs_w_align_policy_compare.png`
- Updated interpretation:
  - optimal `w_align` is pressure-dependent under predictive pursuer;
  - low pressure (`sr≈0.9/1.1`) prefers moderate alignment (`w≈0.35~0.40`);
  - higher pressure (`sr≈1.3/1.4`) prefers low alignment (`w≈0.10~0.12`);
  - high alignment (`w>=0.5`) remains consistently poor.
- Reader-facing synthesis for this batch:
  - `doc/实验结果-对齐比例扩展实验与策略对照图-20260207.md`.

## Criticality relation under pursuer-policy change (matched full-grid)

- Matched full-grid comparison completed (same `w_align` grid and seeds for both policies):
  - `p1`: `runs/sweep_20260207_155527_grid` -> `doc/results_20260207_walign_share_noise0_p1intercept_sr09111314_120seeds_full/`
  - `p0`: `runs/sweep_20260207_161229_grid` -> `doc/results_20260207_walign_share_noise0_p0nearest_sr09111314_120seeds_full/`
- Combined analysis outputs:
  - `doc/results_20260207_criticality_under_policy_switch/criticality_policy_summary.csv`
  - figures:
    - `safe_vs_walign_policy_switch.png`
    - `safe_vs_chi_policy_switch.png`
    - `corr_safe_chi_and_chilocal_policy_switch.png`
    - `near_critical_gain_chi_vs_chilocal_policy_switch.png`
- Stable interpretation:
  1. With global `chi` as proxy, near-critical benefit is **not** supported; `corr(safe,chi)` is negative across all speed layers and becomes more negative under `p1_intercept`.
  2. With local `chi_local` as proxy, near-critical benefit is conditionally supported; under `p1_intercept`, `corr(safe,chi_local)` stays positive and high-vs-low quartile gain remains positive.
  3. Therefore, in this task family, conclusions about “criticality helps” are proxy-dependent and must be reported with explicit metric definitions.
- Reader-facing report:
  - `doc/实验结果-追捕策略改变后临界态与逃跑收益-20260207.md`.
