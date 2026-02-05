# Fixed pursuer-count scan (Np=2): speed ratio × w_align

## Experiment setup

- Pursuer count is fixed by the base config.

- Aggregation: mean ± 95% CI across seeds for each (v_p/v_e, w_align).

## Artifacts

- Sweep directory: `runs\sweep_20260205_135921_grid`

- Base config: `runs\sweep_20260205_135921_grid\base_config.json`

- Group summary (aggregated): `doc\results_20260205_fixedNp_scan\group_summary.csv`

- Figures: `doc\results_20260205_fixedNp_scan\figs`


## Aggregated summary

| v_p/v_e | best w_align (by safe_frac mean) | safe_frac mean |

|---:|---:|---:|

| 1 | 0.3 | 0.3055 |

| 1.1 | 0.4 | 0.2781 |

| 1.2 | 0.4 | 0.2570 |


## Plots

![safe_vs_w_align](figs/safe_vs_w_align.png)

![captured_vs_w_align](figs/captured_vs_w_align.png)

![P_mean_vs_w_align](figs/P_mean_vs_w_align.png)

![heatmap_safe_mean](figs/heatmap_safe_mean.png)
