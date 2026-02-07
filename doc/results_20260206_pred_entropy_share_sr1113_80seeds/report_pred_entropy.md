# Predictive entropy (share mode, sr=1.1/1.3, seeds=80)

## Setup

- Sweep directory: `runs/sweep_20260206_214125_grid`

- Bins: `72`

- EMA span: `10`

- N-gram order: `2`

- Valid runs used: `3320`


## Per-speed correlations (group means)

| v_p/v_e | corr(safe, Var(H_pred)) |

|---:|---:|

| 1.1 | -0.164 |

| 1.3 | 0.559 |


- pooled corr(safe, Var(H_pred)) across all grouped points: `0.132`


## Plots

![pred_entropy_var_vs_w_align](figs/pred_entropy_var_vs_w_align.png)

![safe_vs_w_align](figs/safe_vs_w_align.png)

![scatter_safe_vs_pred_entropy_var](figs/scatter_safe_vs_pred_entropy_var.png)

![heatmap_pred_entropy_var_mean](figs/heatmap_pred_entropy_var_mean.png)

## Raw (non-mean) plots

![raw_scatter_safe_vs_pred_entropy_var](figs/raw_scatter_safe_vs_pred_entropy_var.png)

![raw_scatter_safe_vs_pred_entropy_var_by_w](figs/raw_scatter_safe_vs_pred_entropy_var_by_w.png)

![raw_scatter_safe_vs_pred_entropy_var_sr11](figs/raw_scatter_safe_vs_pred_entropy_var_sr11.png)

![raw_scatter_safe_vs_pred_entropy_var_sr13](figs/raw_scatter_safe_vs_pred_entropy_var_sr13.png)
