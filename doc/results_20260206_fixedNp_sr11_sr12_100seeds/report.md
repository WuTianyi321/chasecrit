# Fixed pursuer-count scan (speed_ratio=1.1,1.2; seeds=100; steps=600)

## Experiment setup

- Pursuer count is fixed by the base config.

- Aggregation: mean ± 95% CI across seeds for each (v_p/v_e, w_align).

## Artifacts

- Sweep directory: `runs/sweep_20260206_132212_grid`

- Base config: `runs/sweep_20260206_132212_grid/base_config.json`

- Group summary (aggregated): `doc/results_20260206_fixedNp_sr11_sr12_100seeds/group_summary.csv`

- Figures: `doc/results_20260206_fixedNp_sr11_sr12_100seeds/figs`


## Aggregated summary

| v_p/v_e | best w (safe) | safe | best w (χ) | χ | best w (χ_local) | χ_local | best w (τ) | τ | best w (ξ) | ξ |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| 1.1 | 0.7 | 0.3725 | 0.2 | 4.8432 | 0 | 4.4615 | 0.45 | 199.0000 | 0.95 | 26.6500 |

| 1.2 | 0.4 | 0.3416 | 0.45 | 4.5407 | 0 | 2.9908 | 0.5 | 198.3523 | 0.85 | 25.9000 |


## Criticality–performance relationships (group means)

| v_p/v_e | corr(safe, χ) | |w_safe-w_χ| | corr(safe, χ_local) | |w_safe-w_χ_local| | corr(safe, τ) | |w_safe-w_τ| | corr(safe, ξ) | |w_safe-w_ξ| |

|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| 1.1 | 0.366 | 0.500 | -0.433 | 0.700 | 0.345 | 0.250 | 0.407 | 0.250 |

| 1.2 | 0.432 | 0.050 | -0.484 | 0.400 | 0.451 | 0.100 | 0.464 | 0.450 |


## Plots

![safe_vs_w_align](figs/safe_vs_w_align.png)

![captured_vs_w_align](figs/captured_vs_w_align.png)

![P_mean_vs_w_align](figs/P_mean_vs_w_align.png)

![chi_vs_w_align](figs/chi_vs_w_align.png)

![chi_local_vs_w_align](figs/chi_local_vs_w_align.png)

![tau_P_ar1_vs_w_align](figs/tau_P_ar1_vs_w_align.png)

![heatmap_safe_mean](figs/heatmap_safe_mean.png)

![heatmap_chi_mean](figs/heatmap_chi_mean.png)

![scatter_safe_vs_chi](figs/scatter_safe_vs_chi.png)

![scatter_safe_vs_chi_local](figs/scatter_safe_vs_chi_local.png)

![scatter_safe_vs_xi](figs/scatter_safe_vs_xi.png)

![scatter_safe_vs_tau_P_ar1](figs/scatter_safe_vs_tau_P_ar1.png)

![scatter_safe_vs_components](figs/scatter_safe_vs_components.png)
