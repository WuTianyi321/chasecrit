# ChaseCrit

Research sandbox for swarm near-criticality in pursuit–evasion tasks (focus on evader swarm).

## Quickstart (uv)

- Create/update env: `uv sync`
- Run one episode (defaults): `uv run chasecrit --seed 0 --steps 800`
- Run with config: `uv run chasecrit run --config configs/base.toml --seed 0 --steps 800`
- Sweep `w_align` only: `uv run chasecrit sweep-w-align --values 0 0.2 0.4 0.6 0.8 1.0 --seeds 0 1 2 3 4 --steps 400 --no-save-events --no-save-timeseries`
- Sweep grid `v_p/v_e × w_align` (fixed `N_p`): `uv run chasecrit sweep-grid --speed-ratio 1.0 1.1 1.2 --w-align 0 0.1 0.2 0.3 0.4 0.5 --seeds 0 1 2 3 4 --steps 400 --no-save-events --no-save-timeseries`
- Single-parameter alignment-share route (noise=0): use `configs/walign_share_noise0.toml`, then sweep `--w-align`; in this mode `w_align` is the alignment share in `[0,1]`.
- Predictive-entropy report (requires per-run timeseries): run sweep with `--save-runs` and without `--no-save-timeseries`, then `uv run chasecrit report-pred-entropy --sweep-dir runs/<sweep_dir> --out-dir doc/results_<name> --title "..."`
- Phase identification (no pursuers/zones): `uv run chasecrit phase-sweep-noise --noise 0 0.2 0.4 0.6 0.8 1.0 --seeds 0 1 2 3 4 --steps 1200 --jobs 8`
- Task noise sweep: `uv run chasecrit sweep-noise --speed-ratio 1.0 1.1 1.2 --noise 0 0.2 0.4 0.6 --seeds 0 1 2 3 4 --steps 600 --w-align 1.0 --no-save-events --no-save-timeseries --jobs 8`
- Generate plots + markdown report from a sweep: `uv run chasecrit report --sweep-dir runs/<sweep_dir> --out-dir doc/results_<name> --title "..."` 

Outputs are written under `runs/` (each run has `config.json`, `summary.json`, and optionally `timeseries.csv` + `events.json`).

## Repository layout

- `src/chasecrit/`: simulation core, policies, metrics, CLI
- `configs/`: baseline experiment configs
- `tests/`: regression/equivalence/benchmark sanity tests
- `runs/`: raw sweep/run artifacts (not version-controlled)
- `doc/`: reader-facing experiment design/results and report artifacts
- `memory/`: long-term project memory and daily logs

## Paper prep entrypoint

For the current manuscript-oriented summary and reproducibility checklist, see:

- `doc/论文准备-结果索引与复现清单.md`
