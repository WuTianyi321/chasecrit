# 方法细节：SOC-v4-VarH 与前瞻追捕（p1_intercept）

## 1. 目的与范围

本文档给出当前实现中“自组织临界-追逃融合算法”的完整计算细节，覆盖：

1. 逃跑者控制律（含 SOC-v4-VarH）。
2. 前瞻追捕算法 `p1_intercept`。
3. 二者在单步仿真中的耦合顺序。

实现对齐文件：

- `src/chasecrit/policies.py`
- `src/chasecrit/sim.py`
- `src/chasecrit/config.py`

## 2. 状态与记号

- 逃跑者位置/速度：`pos_e ∈ R^{N_e×2}`, `vel_e ∈ R^{N_e×2}`。
- 追捕者位置/速度：`pos_p ∈ R^{N_p×2}`, `vel_p ∈ R^{N_p×2}`。
- 活跃逃跑者掩码：`alive_mask = alive & (~safe) & (~captured)`。
- 世界尺寸：`world_size = [Lx, Ly]`，边界模式 `periodic` 或 `reflecting`。
- 步长：`dt`。

SOC 状态（每个逃跑者 i）：

- `stress_i`：应激状态。
- `align_share_i`：对齐占比，限制在 `[soc_align_min, soc_align_max]`。
- `pred_prev_bin_i`：上一时刻方向离散桶。
- `pred_trans_counts_i[b_prev,b_cur]`：方向转移计数。
- `pred_entropy_ema_i`：预测熵 EMA。

## 3. 单步仿真顺序（`Simulator.step`）

1. 安全区移动与刷新（G0 机制）。
2. 计算 `alive_mask`。
3. 更新逃跑者速度：`evader_step(...)`。
4. 更新逃跑者位置并处理边界。
5. 更新追捕者速度：`p0_nearest` 或 `p1_intercept`。
6. 更新追捕者位置并处理边界。
7. 瞬时捕获判定（距离阈值 `capture_radius`）。
8. 进入安全区判定与容量处理。
9. 记录 timeseries 与 SOC 指标。

## 4. 逃跑者基础控制（SOC 前）

### 4.1 邻居对齐项 `d_align`

- 在邻域半径 `r_nbr` 内聚合邻居单位速度方向，归一化得到 `d_align`。
- 小规模群体可走 dense/numba 路径，大规模走网格分桶近邻路径；语义一致。

### 4.2 分离项 `d_sep`

- 在 `r_sep` 内按距离倒数平方近似加权斥力：
  \[
  s_i = -\sum_{j\in\mathcal N_{sep}(i)} \frac{\Delta x_{ij}}{\max(\|\Delta x_{ij}\|^2,\varepsilon)}
  \]
- 归一化后乘 `sep_strength` 得到 `d_sep`。

### 4.3 规避追捕项 `d_avoid`

- 在 `r_pred` 内对追捕者做反向加权和，归一化得到 `d_avoid`。

### 4.4 目标项 `d_goal` 与探索项 `d_explore`

- 在 `r_detect` 内可检测到有效安全区时，取最近安全区方向为 `d_goal`。
- 无可见安全区时使用 `d_explore`（当前朝向；接近静止时随机方向）。

### 4.5 非对齐合成方向

\[
 d_{non\_align} = \operatorname{unit}\Big(
 w_{avoid} d_{avoid}
 + w_{goal}(\mathbf{1}_{goal} d_{goal})
 + w_{explore}(1-\mathbf{1}_{goal}) d_{explore}
 + d_{sep}
 \Big)
\]

## 5. 预测熵与 Var(H) 计算（SOC v3/v4 共用）

对每个活跃逃跑者 i：

1. 将当前决策方向角离散到 `bins=soc_heading_bins` 个桶：`curr_bin_i`。
2. 读取上一桶 `prev_bin_i` 对应的转移行向量 `row_i`。
3. 平滑概率（`soc_heading_smoothing`）：
   \[
   p_{i,k}=\frac{row_{i,k}+\alpha}{\sum_j row_{i,j}+\alpha\cdot bins}
   \]
4. 归一化熵：
   \[
   H_i = \frac{-\sum_k p_{i,k}\log_2 p_{i,k}}{\log_2(bins)}
   \]
5. 熵波动（相对 EMA）：
   \[
   \Delta H_i = |H_i - EMA_i|,
   \quad EMA_i \leftarrow \beta H_i + (1-\beta)EMA_i
   \]
   其中 `β = soc_entropy_ema_alpha`。
6. 更新转移计数：上一行先按 `soc_heading_decay` 衰减，再给 `(prev_bin_i, curr_bin_i)` 加 1。

群体层统计：

- `pred_entropy_var = Var({H_i})`。
- `entropy_fluct_mean = mean({ΔH_i})`。

## 6. SOC-v4-VarH 控制律

模式：`soc_mode = "v4_varh"`。

### 6.1 先验应激通道（可开可关）

所有 SOC 模式共享应激更新：

\[
stress_i \leftarrow (1-soc\_stress\_decay)\,stress_i
+ soc\_surprise\_gain\cdot surprise_i
+ soc\_threat\_gain\cdot threat_i
+ soc\_entropy\_gain\cdot \Delta H_i
\]

- `surprise_i = 0.5(1 - \langle heading_i, d_i\rangle)`。
- `threat_i` 由最近追捕者距离归一化得到。
- 若开启 `soc_neighbor_coupling`，topple 触发前会追加邻居耦合 bump。

### 6.2 topple 通道（可开可关）

- 若 `stress_i > soc_threshold` 则 topple：
  - `stress_i -= soc_release`（下限 0）
  - 可选角噪声扰动 `soc_topple_noise`
  - 可选 `align_share_i -= soc_align_drop`

### 6.3 Var(H) 反馈主环

定义误差：
\[
 e = soc\_varh\_target - pred\_entropy\_var
\]

若 `|e| > soc_varh_deadband`，更新：
\[
align\_share_i \leftarrow align\_share_i - soc\_varh\_gain\cdot e
\]

再执行：

- 可选回归基准 `w_align`：
  \[
  align\_share_i \leftarrow align\_share_i + soc\_align\_relax\,(w_{align}-align\_share_i)
  \]
  （由 `soc_relax_to_w_align` 控制）
- 最终裁剪到 `[soc_align_min, soc_align_max]`。

### 6.4 方向合成与速度更新

在 SOC 开启时：
\[
 d_i = \operatorname{unit}(align\_share_i\, d_{align,i} + (1-align\_share_i)\, d_{non\_align,i})
\]

再加可选角噪声 `angle_noise`，并以惯性更新速度：
\[
 v_i^{next} = (1-inertia)\,v_i + inertia\cdot v_e\,d_i
\]

## 7. 前瞻追捕算法 `p1_intercept`

模式：`pursuers.policy = "p1_intercept"`。

### 7.1 截获时间求解

对每个追捕者 p、每个活跃逃跑者 e：

- 相对位置：`r = x_e - x_p`（周期边界下先做最短镜像差）。
- 目标速度：`v = v_e`。
- 追捕速度标量：`v_p`。

假设目标匀速，截获条件：
\[
\|r + v t\| = v_p t
\]

展开为二次方程：
\[
(v\cdot v - v_p^2)t^2 + 2(r\cdot v)t + (r\cdot r)=0
\]

即：`a t^2 + b t + c = 0`。

- 线性近似分支：`|a|<eps`，用 `t=-c/b`。
- 二次分支：判别式 `disc=b^2-4ac`，取最小正根；无正根则记为无解。

### 7.2 目标选择与瞄准点

- 在所有可解目标中选 `t` 最小者。
- 若 `intercept_tmax > 0`，用 `t_eff = min(t, intercept_tmax)`。
- 预测瞄准点方向：`aim_pred = r + v * t_eff`。

### 7.3 预测强度混合（关键）

`intercept_gain ∈ [0,1]`：
\[
aim = (1-intercept\_gain)\,r + intercept\_gain\,aim_{pred}
\]

- `intercept_gain=0`：退化到直接追当前位置。
- `intercept_gain=1`：纯前瞻拦截。

若所有目标都无解，退化为最近目标追击（`argmin ||r||^2`）。

最终追捕速度：
\[
v_p^{next} = v_p \cdot \operatorname{unit}(aim)
\]

## 8. 融合算法中的关键耦合

1. `p1_intercept` 越强（高 `intercept_gain` 或长预测时域），逃跑群体面临更强“可预测性惩罚”。
2. `SOC-v4-VarH` 通过调节 `align_share` 控制群体行为可预测性波动（`Var(H)`）。
3. 因此，核心研究关系是三元耦合：
   - 追捕预测能力（`intercept_gain`, `intercept_tmax`）
   - 逃跑侧 `Var(H)`（`soc_pred_entropy_var`）
   - 任务收益（`safe_frac`）

## 9. 参数字典（与配置字段一致）

### 9.1 SOC-v4-VarH 关键参数（`[evaders]`）

- `soc_enabled`：是否启用 SOC。
- `soc_mode`：应为 `"v4_varh"`。
- `soc_varh_target`：Var(H) 目标值。
- `soc_varh_gain`：Var(H) 误差到 `align_share` 的反馈增益。
- `soc_varh_deadband`：误差死区。
- `soc_relax_to_w_align`、`soc_align_relax`：是否向基准 `w_align` 回归及回归速度。
- `soc_align_min`、`soc_align_max`：`align_share` 限幅。

可选并行通道（用于混合机制研究）：

- `soc_surprise_gain`、`soc_threat_gain`、`soc_entropy_gain`（应激注入）
- `soc_threshold`、`soc_release`、`soc_neighbor_coupling`（topple）
- `soc_align_drop`、`soc_topple_noise`（topple 后对齐与方向扰动）

### 9.2 前瞻追捕关键参数（`[pursuers]`）

- `policy = "p1_intercept"`
- `intercept_gain`：前瞻混合权重。
- `intercept_tmax`：前瞻时域上限（`<=0` 表示不截断）。
- `speed_ratio`：`v_p / v_e`。
- `capture_radius`：瞬时捕获阈值。

## 10. 与日志指标的对应关系

- `soc_pred_entropy_var`：每步群体 `Var(H)`，summary 中为时间均值。
- `soc_entropy_fluct_mean`：每步 `mean(|H-EMA(H)|)`，summary 中为时间均值。
- `safe_frac`：最终进入安全区比例。
- `chi`、`chi_local`：任务内涨落代理指标。

上述指标用于回答“预测能力变化下，Var(H) 与生存率如何变化”的实验问题。
