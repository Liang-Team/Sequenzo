# Sequenzo vs TraMineR 功能对比分析

## 概述
本文档对比 Sequenzo 和 TraMineR 的功能，识别 Sequenzo 中缺失的功能。

**TraMineR 版本**: 2.2-13 (2025-12-14)  
**对比日期**: 2026-04-30

---

## 1. 序列数据定义和格式转换

### ✅ Sequenzo 已实现
- `SequenceData` - 定义状态序列对象
- 数据格式支持（宽格式、长格式）
- 缺失值处理

### ❌ Sequenzo 缺失的功能

#### 1.1 事件序列支持
- **`seqecreate()`** → `create_event_sequences()`（已实现）
- **`seqelist`** → `EventSequenceList`（已实现）
- **`eseq`** → `EventSequence`（已实现）
- **`is.eseq()` / `is.seqelist()`** - 仅缺同名便捷检查函数；当前可用 `isinstance(x, EventSequence)` / `isinstance(x, EventSequenceList)` 等价实现
- **`seqelength()`** - 已可通过 `EventSequenceList.lengths`（以及单序列 `len(EventSequence)`）获取
- **`seqeweight()`** - 已可通过 `EventSequenceList.weights` 获取

#### 1.2 格式转换功能
- **`seqformat()`** - 更全面的格式转换功能：
  - STS ↔ SPELL 转换
  - STS ↔ TSE (Time-Stamped Event) 转换
  - STS ↔ SRS (Sequence of Repeated States) 转换
  - SPS (State-Period-Sequence) 格式支持
  - DSS (Distinct State Sequence) 格式转换
- **`seqdecomp()`** → `sequenzo.sequence_operations.seqdecomp()`（已实现）
- **`seqconc()`** → `sequenzo.sequence_operations.seqconc()`（已实现）
- **`seqsep()`** → `sequenzo.sequence_operations.seqsep()`（已实现）

#### 1.3 序列操作
- **`seqrecode()`** → `sequenzo.sequence_operations.seqrecode()`（已实现）
- **`seqshift()`** → `sequenzo.sequence_operations.seqshift()`（已实现）
- **`seqasnum()`** → `sequenzo.sequence_operations.seqasnum()`（已实现）

---

## 2. 序列可视化

### ✅ Sequenzo 已实现
- `plot_sequence_index()` - 序列索引图
- `plot_most_frequent_sequences()` - 最频繁序列图
- `plot_state_distribution()` - 状态分布图
- `plot_mean_time()` - 平均时间图
- `plot_modal_state()` - 模态状态图
- `plot_relative_frequency()` - 相对频率图
- `plot_transition_matrix()` - 转换矩阵图
- `plot_single_medoid()` - 单个中位数图

### ❌ Sequenzo 缺失的可视化功能

#### 2.1 高级绘图类型
- **`seqpcplot()`** - 平行坐标图 (Parallel Coordinate Plot)
- **`seqHtplot()`** - 熵指数图 (Entropy Index Plot)
- **`seqdHplot()`** - 带熵的分布图 (Distribution plot with entropy overlay)
- **`seqrfplot()`** - RF组中位数序列图 (RF group medoid sequences plot)
- **`seqplotMD()`** - 多域/多通道序列图 (Multi-domain/multichannel sequence plot)

#### 2.2 绘图增强功能
- **`seqlegend()`** - 独立的图例函数（支持更多自定义选项）
- **`seqgbar()`** - 序列条形图
- **`seqmaintokens()`** - 显示最频繁的tokens

#### 2.3 绘图选项
- 支持 `tick.last` 参数（在时间轴最后位置显示刻度）
- 支持 `sampling` 和 `sample.meth`（大序列集的子采样绘图）
- 支持 `bar.labels`（条形图标签）
- 支持 `info` 参数（控制是否显示统计信息）

---

## 3. 序列特征/指标

### ✅ Sequenzo 已实现
- `get_sequence_length()` - 序列长度
- `get_spell_durations()` - spell持续时间
- `get_visited_states()` - 访问的状态
- `get_recurrence()` - 递归性
- `get_mean_spell_duration()` - 平均spell持续时间
- `get_duration_standard_deviation()` - 持续时间标准差
- `get_within_sequence_entropy()` - 序列内熵
- `get_cross_sectional_entropy()` - 横截面熵
- `get_turbulence()` - 湍流指数
- `get_complexity_index()` - 复杂度指数
- `get_volatility()` - 波动性指数
- `get_integration_index()` - 整合指数
- `get_entropy_difference()` - 熵差
- `get_spell_duration_variance()` - spell持续时间方差
- `get_badness_index()` - 不良指数
- `get_degradation_index()` - 退化指数
- `get_precarity_index()` - 不稳定指数
- `get_insecurity_index()` - 不安全感指数
- `get_mean_time_in_states()` - 各状态平均时间
- `get_modal_state_sequence()` - 模态状态序列
- `get_positive_negative_indicators()` - 正负指标

### ❌ Sequenzo 缺失的指标

#### 3.1 综合指标函数
- **`seqindic()`** - 综合指标函数，可以一次性计算多个指标：
  - 支持指标组：`'basic'`, `'diversity'`, `'complexity'`, `'binary'`, `'ranked'`
  - 支持单个指标：`'degrad'`, `'bad'`, `'prec'`, `'insec'`, `'recu'`, `'nvolat'`, `'meand'`, `'dustd'`, `'meand2'`, `'dustd2'`, `'turb2'`, `'turb2n'`

#### 3.2 序列统计函数
- **`seqstatd()`** - 状态分布统计（更全面的版本）
- **`seqstatf()`** - 序列频率统计
- **`seqstatl()`** - 序列长度统计
- **`seqnum()`** - 序列数量统计
- **`seqsubsn()`** - 已有基础实现（`sequence_characteristics/simple_characteristics.py`），但与 TraMineR 仍有参数与输出细节差异

#### 3.3 转换相关指标
- **`seqtransn()`** - 转换次数（归一化版本）
- **`seqtrate()`** - 转换率（更完整的实现，支持 `lag` 参数）
- **`seqmpos()`** - 最频繁位置

#### 3.4 其他指标
- **`seqlogp()`** - 序列概率对数（支持 `begin='global.freq'`）
- **`seqST()`** - 湍流指数（更完整的实现，支持 `type` 参数选择持续时间方差类型）
- **`seqipos()`** - 积极状态比例（支持 `index` 参数选择指标类型）

---

## 4. 距离度量

### ✅ Sequenzo 已实现
- `get_distance_matrix()` 支持的方法：
  - OM (Optimal Matching)
  - OMloc, OMslen, OMspell, OMspellUnitFree, OMtspell, OMstran
  - HAM (Hamming)
  - DHD (Dynamic Hamming Distance)
  - CHI2 (Chi-squared)
  - EUCLID (Euclidean)
  - LCS (Longest Common Subsequence)
  - LCP (Longest Common Prefix)
  - RLCP (Reversed Longest Common Prefix)
  - LCPspell, RLCPspell
  - NMS (Number of Matching Subsequences)
  - NMSMST (NMS weighted by Minimum Shared Time)
  - SVRspell (Subsequence Vectorial Representation)
  - TWED (Time Warp Edit Distance)

### ❌ Sequenzo 缺失的距离度量

#### 4.1 距离方法
- （当前已无明显遗漏；`OMloc` 与 `OMtspell` 已在 `get_distance_matrix()` 中实现）

#### 4.2 距离相关功能
- （已实现，TraMineR 语义对齐）
  - **`seqalign()`** - 序列对齐细节（返回操作序列、逐步成本和动态规划矩阵）
  - **`seqfind()`** - 查找序列（返回在目标序列集中的出现位置，1-based）
  - **`seqLLCP()`** - 最长公共前缀长度（两个序列之间）
  - **`seqLLCS()`** - 最长公共子序列长度（两个序列之间）

---

## 5. 聚类和代表性序列

### ✅ Sequenzo 已实现
- `KMedoids` - K-medoids聚类
- `Cluster` - 层次聚类
- `clara()` - CLARA算法（大数据集）
- `plot_single_medoid()` - 单个中位数可视化

### ❌ Sequenzo 缺失的功能

#### 5.1 代表性序列提取
- （已实现，核心计算层可调用）
  - **`get_representative_sequences()`**（TraMineR: `seqrep()`）- 提取代表性序列集（支持 `density` / `freq` / `dist` / `random` 标准）
  - **`get_relative_frequency_representatives()`**（TraMineR: `seqrf()`）- Relative Frequency（RF）分组代表序列提取（可供可视化复用）
  - **`get_representative_objects()`**（TraMineR: `dissrep()`）- 从距离矩阵提取代表性对象
  - **`get_relative_frequency_groups()`**（TraMineR: `dissrf()`）- 基于距离矩阵的 RF 分组代表对象提取
  - **`get_distance_center()`**（TraMineR: `disscenter()`）- 距离中心/组中位数计算

#### 5.2 代表性序列可视化
- **`seqrplot()`** - 代表性序列图
- **`seqrfplot()`** - RF组中位数序列图

---

## 6. 事件序列分析

### ✅ Sequenzo 已基本实现（与 TraMineR 对齐）

#### 6.1 事件序列创建和处理
- **`seqecreate()`** → `create_event_sequences()`（位于 `sequenzo/with_event_history_analysis/event_sequence.py`）
  - 支持从 STS（`SequenceData`）或 TSE 数据框创建事件序列对象
- **`seqelist` / `eseq` / `subseqelist`** → `EventSequenceList`, `EventSequence`, `SubsequenceList`
  - 保存 ID、时间戳、事件字典、权重、长度等信息
- **`seqeid()` / `seqelength()` / `seqeweight()`**  
  - ID：`EventSequence.id`
  - 长度：`EventSequenceList.lengths`
  - 权重：`EventSequenceList.weights`
- **`seqeapplysub()`** → `count_subsequence_occurrences()`
  - 通过 `EventSequenceConstraint.count_method` 支持 COBJ / CDIST_O / CWIN / CMINWIN / CDIST 等计数方式
- **`seqeconstraint()`** → `EventSequenceConstraint`
  - 与 TraMineR 类似的时间窗、最大间隔、计数方法等控制选项

#### 6.2 事件子序列挖掘
- **`seqefsub()`** → `find_frequent_subsequences()`
  - 支持通过 `min_support` 或 `pmin_support` 设定频繁子序列阈值
  - 支持用户指定子序列字符串（与 TraMineR 子序列语法兼容，如 `"(A)-(B,C)"`）
- **`seqecmpgroup()`** → `compare_groups()`
  - 使用卡方检验（可选 Bonferroni 校正）比较组间子序列分布差异
  - 返回的统计结果与 TraMineR 输出结构对应（p-value, statistic, group-specific frequencies/residuals）
- **`seqecontain()`** → `check_event_subsequence_containment()`
  - 内部基于 `_find_subsequence_presence()`，对每条序列返回是否包含指定子序列的布尔向量
  - 支持字符串子序列和 `EventSequence` 两种输入形式

#### 6.3 事件序列转换
- **`seqe2tse()`** → `convert_event_sequences_to_tse()`
  - 将 `EventSequenceList` 转换为 TSE 数据框，列名为 `id`, `timestamp`, `event`
  - 同一 ID 内按时间戳和事件名排序，结构与 TraMineR `seqe2tse()` 输出保持一致
- **`seqetm()`** → `compute_event_transition_matrix()`
  - 在所有事件序列上计算事件转换矩阵：
    - 行：起始事件
    - 列：目标事件
    - 元素：加权转移计数或转移概率（按行归一化）
  - `weighted=True` 时使用 `EventSequenceList.weights` 作为权重

#### 6.4 事件序列可视化
- **`plot.subseqelist()` / `plot.subseqelistchisq()`**  
  - 在 `event_sequence_visualization.py` 中提供等价的可视化函数，用于：
    - 展示频繁子序列的支持度 / 计数
    - 展示组间区分性子序列的卡方统计结果
  - 输出内容与 TraMineR 相同，只是接口形式采用 Python 风格函数而非 S3 方法

---

## 7. 树分析

### ✅ Sequenzo 已基本实现（基于距离矩阵的树分析）

#### 7.1 序列回归树
- **`seqtree()`** → `build_sequence_tree()`（位于 `sequenzo/tree_analysis/seqtree.py`）
  - 从 `SequenceData` 提取或计算距离矩阵，再调用距离树引擎
- **`seqtreedisplay()`** → `plot_tree()` / `print_tree()`（位于 `tree_visualization.py`）
- **`seqtree2dot()`** → `export_tree_to_dot()`（同上，导出 GraphViz DOT）

#### 7.2 距离树
- **`disstree()`** → `build_distance_tree()`（位于 `sequenzo/tree_analysis/disstree.py`）
- **`disstreedisplay()`** → `plot_tree()` / `print_tree()`（与序列树共用）
- **`disstree2dot()` / `disstree2dotp()`** → `export_tree_to_dot()`  
  - 支持带参数导出（如控制节点信息、标签等）
- **`disstreeleaf()`** → `_get_leaf_memberships()` + 公共封装 `get_leaf_membership()`
- **`disstree.get.rules()`** → `get_classification_rules()`
- **`disstree.assign()`** → `assign_to_leaves()`

---

## 8. 多域分析

### ✅ Sequenzo 已实现
- `create_idcd_sequence_from_csvs()` - IDCD序列创建
- `compute_cat_distance_matrix()` - CAT距离矩阵
- `compute_dat_distance_matrix()` - DAT距离矩阵
- `get_interactive_combined_typology()` - 交互式组合类型
- `get_association_between_domains()` - 域间关联
- `linked_polyadic_sequence_analysis()` - 链接多体序列分析

### ❌ Sequenzo 缺失的功能

#### 8.1 多域序列处理
- **`seqMD()`** - 多域序列对象（更完整的实现）
  - 支持 `with.missing` 向量（每个域不同值）
  - 支持 `fill.with.miss` 参数（用缺失值填充较短序列）

#### 8.2 多域可视化
- **`seqplotMD()`** - 多域/多通道序列图（按域和组绘图）

#### 8.3 域间关联分析
- **`seqdomassoc()`** → `get_association_between_domains()`（已实现主要统计框架）
  - 当前仍建议核对与 TraMineR 的逐参数语义一致性（如细粒度选项与默认值）

---

## 9. 距离矩阵分析

### ✅ Sequenzo 已部分实现

#### 9.1 距离矩阵统计
- **`dissvar()`** → `compute_pseudo_variance()`（位于 `sequenzo/tree_analysis/tree_utils.py`）
- **`dissassoc()`** → `compute_distance_association()`（同上）
- **`dissmfacw()`** → `dissmfacw()`（同上，新增）
  - 对多个因子（列）逐个调用 `compute_distance_association()`，返回每个因子的 Pseudo R² / Pseudo F / p-value 等汇总表
- **`dissmergegroups()`** → `dissmergegroups()`（同上，新增）
  - 从初始分组出发，迭代合并“对 Pseudo R² 损失最小”的两个组，直到达到目标组数
  - 返回合并历史和最终分组结果，便于重现分组合并路径

#### 9.2 距离矩阵可视化
- **`dissrep()`** / **`dissrf()`**  
  - 目前尚无完全同名的高层封装；代表性对象和 RF 中位数主要通过：
    - K-medoids / 层次聚类 / CLARA（`sequenzo/clustering`）
    - `plot_relative_frequency()` + `disscentertrim()`（`sequenzo/visualization/plot_relative_frequency.py` 和 `clustering/utils/disscenter.py`）实现
  - 若希望 1:1 接口对应 TraMineR，后续可以基于上述内部函数封装 `dissrep()` / `dissrf()` 名称

---

## 10. 序列差异分析

### ✅ Sequenzo 已实现

- **`seqdiff()`** → `compare_groups_across_positions()`（位于 `sequenzo/compare_differences/seqdiff.py`）
  - 按时间位置做滑动窗口的差异分析，内部使用 `get_distance_matrix()` + `compute_distance_association()`
  - 输出含有 Pseudo F / Pseudo Fbf / Pseudo R² / Bartlett / Levene 等统计量的表格
- **`print.seqdiff()`** → `print_group_differences_across_positions()`
- **`plot.seqdiff()`** → `plot_group_differences_across_positions()`

---

## 11. 序列生成和模拟

### ❌ Sequenzo 缺失的功能

- **`seqgen()`** - 生成随机序列

---

## 12. 工具函数

### ✅ Sequenzo 已实现
- 数据预处理工具
- 缺失值处理
- 权重支持

### ❌ Sequenzo 缺失的工具函数

#### 12.1 序列检查
- **`seqhasmiss()`** - 检查序列是否有缺失值（计数和识别）
- **`seqfcheck()`** - 序列格式检查
- **`is.stslist()`** - 检查是否为状态序列对象

#### 12.2 序列操作
- **`seqmaintokens()`** - 返回最频繁tokens的索引
- **`seqdss()`** → `get_distinct_state_sequences()`（已实现，用户级公开 API）
- **`seqdur()`** → `get_state_spell_durations()`（已实现，用户级公开 API）

#### 12.3 其他工具
- **`alphabet()`** - 获取/设置字母表（支持 `with.missing` 参数）
- **`cpal()`** - 获取/设置颜色调色板
- **`stlab()`** - 获取/设置状态标签
- **`seqdim()`** - 序列维度
- **`seqtab()`** - 序列频率表（更完整的实现）
- **`read.tda.mdist()`** - 读取TDA距离矩阵文件

---

## 13. 高级功能

### ❌ Sequenzo 缺失的高级功能

#### 13.1 序列对齐
- **`seqalign()`** - 序列对齐（显示对齐细节）
  - `plot.seqalign()` - 对齐可视化
  - `print.seqalign()` - 对齐打印

#### 13.2 序列查找
- **`seqfind()`** - 在序列集中查找序列
- **`seqfpos()`** - 首次出现位置
- **`seqfposend()`** - 首次spell结束位置
- **`seqipos()`** - 位置索引（更完整的实现）

#### 13.3 序列比较
- **`seqcomp()`** - 序列比较
- **`seqsubm()`** - 子序列匹配（可能已部分实现）

---

## 14. 权重和统计增强

### ✅ Sequenzo 已实现（统计模块与用户级 API）

#### 14.1 加权统计
- **`weighted.mean()`** → `get_weighted_mean()`（用户级 API，位于 `sequenzo/statistics/weighted.py`）
- **`weighted.var()`** → `get_weighted_variance()`（同上）
- **`weighted.fivenum()`** → `get_weighted_five_number_summary()`（同上）
- 同时保留底层实现：`sequenzo/utils/weighted_stats.py`，用于内部计算复用

#### 14.2 统计增强
- **`seqmeant()`** → `get_mean_time_by_state()`（支持 `show_standard_error` 参数显示标准误差）
- **`seqistatd()`** → `get_individual_state_distribution()`（支持 `as_proportion` 参数计算比例）
- **汇总统计（小白友好）**：
  - `get_sequence_length_summary()`（长度的 count/mean/median/q1/q3 等汇总）
  - `get_transition_count_summary()`（转变次数的 count/mean/median/q1/q3 等汇总）

---

## 15. TraMineRextras 对比（新增）

### ✅ Sequenzo 已部分覆盖 / 有近似能力
- `dissindic()` / `dissvar.grp` 相关能力可由 `compute_distance_association()`、`compute_pseudo_variance()`、`dissmfacw()`、`dissmergegroups()` 组合实现。
- `seqCompare` / `dissCompare` 相关“组间比较”场景，可由 `compare_groups_overall()` 与 `compare_groups_across_positions()` 覆盖核心需求。
- `seqsamm` 家族在 Sequenzo 中有完整实现（并且是重点能力）。

### ❌ 目前仍缺失（TraMineRextras 主要未覆盖项）
- 事件序列扩展：`seqeformat`, `seqedist`, `seqedplot`, `seqerulesdisc`, `seqentrans`, `seqe2stm`
- 序列增强分析：`seqindic.dyn`, `seqsurv`, `seqimplic`, `seqauto`, `seqcta`（TraMineRextras 版本功能面）
- 格式与数据变换：`TSE_to_STS`, `HSPELL_to_STS`, `FCE_to_TSE`, `toPersonPeriod`, `createdatadiscrete`, `convert.g`
- 代表性与可视化扩展：`seqrep.grp`, `seqplot.rf`, `seqplot.tentrop`, `seqsplot`
- 其他工具：`seqgen.missing`, `seqgranularity`, `pamward`, `sortv`, `rowmode`, `group.p`

---

## 总结

### 主要缺失的功能模块（2026-04 更新后）：

1. **代表性序列提取** - 仍部分缺失
   - `seqrep()` 和 `dissrep()` 的统一高层封装
   - RF 组中位数相关的完整 API 家族（当前可通过 K-medoids、`plot_relative_frequency()` 等实现近似功能）

2. **高级可视化** - 部分缺失
   - 平行坐标图 `seqpcplot()`
   - 熵指数图 `seqHtplot()` / 带熵覆盖的分布图 `seqdHplot()`
   - 多域序列图 `seqplotMD()`

3. **综合指标函数** - 缺少统一入口
   - 单个指标（如 `seqlength`, `seqdur`, `recu`, `meand`, `dustd`, `turb`, `prec`, `insec` 等）已在不同模块中实现
   - 但尚未提供类似 TraMineR `seqindic()` 的“一次性计算多个指标”的包装函数

4. **格式转换与工具函数** - 仍有差距
   - 更全面的 `seqformat()` 家族（尤其是 SRS、SPS、DSS 等格式的用户级 API）
   - 部分检查 / 对齐 / 查找函数（如 `seqhasmiss()`, `seqfcheck()`, `seqalign()` 及其可视化家族）目前只在内部或底层工具中部分覆盖

5. **TraMineRextras 扩展功能** - 覆盖面仍有限
   - 事件序列扩展（`seqeformat`, `seqedist`, `seqerulesdisc` 等）
   - 动态指标与扩展可视化（`seqindic.dyn`, `seqplot.rf`, `seqplot.tentrop` 等）

### 建议优先级：

**高优先级**：
1. 代表性序列提取高层 API（`seqrep()`, `dissrep()`, RF 组中位数家族）
2. 综合指标函数统一入口（`seqindic()` 风格的包装函数）

**中优先级**：
3. 高级可视化（平行坐标图、熵图、多域序列图等）
4. 格式转换增强（完善 `seqformat()` 家族接口）
5. TraMineRextras 常用扩展（优先 `seqindic.dyn`, `seqplot.rf`, `seqeformat`）

**低优先级**：
5. 序列生成和模拟（`seqgen()` 等）
6. 工具函数增强（检查 / 对齐 / 查找家族的人性化封装）

---

## 备注

- 本文档基于 TraMineR 2.2-13 版本
- 某些功能可能在 Sequenzo 中已有部分实现但未完全对应 TraMineR 的功能
- 建议根据实际用户需求确定开发优先级

## 特别说明

### Sequenzo 独有的功能（TraMineR 没有）

1. **Prefix Tree 和 Suffix Tree 分析**
   - `build_prefix_tree()` - 前缀树构建
   - `build_suffix_tree()` - 后缀树构建
   - 这些是 Sequenzo 的创新功能，用于分析序列的分支和收敛模式

2. **Hidden Markov Models (HMM)**
   - `HMM`, `MHMM`, `NHMM` - 隐马尔可夫模型
   - TraMineR 不包含 HMM 功能

3. **Event History Analysis (SAMM)**
   - `SAMM` - Sequence Analysis Multi-State Model
   - 注意：这与 TraMineR 的 Event Sequence Analysis 不同

4. **大数据处理**
   - `clara()` - CLARA 算法用于大数据集聚类
   - 更优化的实现

### 功能差异说明

1. **事件序列 vs 事件历史分析**
   - TraMineR 的 `seqecreate()` 等是用于事件序列（Event Sequence）分析
   - Sequenzo 的 `SAMM` 是用于事件历史（Event History）分析
   - 两者概念不同，不能直接对应

2. **树分析**
   - TraMineR 的 `seqtree()` / `disstree()` 是基于距离矩阵的回归树
   - Sequenzo 的 `prefix_tree` / `suffix_tree` 是序列结构树
   - 两者用途不同，不能直接对应
