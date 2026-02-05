# ChaseCrit

Research sandbox for swarm near-criticality in pursuit–evasion tasks (focus on evader swarm).

## Quickstart (uv)

- Create/update env: `uv sync`
- Run one episode (defaults): `uv run chasecrit --seed 0 --steps 800`
- Run with config: `uv run chasecrit run --config configs/base.toml --seed 0 --steps 800`
- Sweep `w_align` only: `uv run chasecrit sweep-w-align --values 0 0.2 0.4 0.6 0.8 1.0 --seeds 0 1 2 3 4 --steps 400 --no-save-events --no-save-timeseries`
- Sweep grid `v_p/v_e × w_align` (fixed `N_p`): `uv run chasecrit sweep-grid --speed-ratio 1.0 1.1 1.2 --w-align 0 0.1 0.2 0.3 0.4 0.5 --seeds 0 1 2 3 4 --steps 400 --no-save-events --no-save-timeseries`
- Generate plots + markdown report from a sweep: `uv run chasecrit report --sweep-dir runs/<sweep_dir> --out-dir doc/results_<name> --title "..."` 

Outputs are written under `runs/` (each run has `config.json`, `summary.json`, and optionally `timeseries.csv` + `events.json`).
