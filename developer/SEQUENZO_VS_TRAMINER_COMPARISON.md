# Sequenzo vs TraMineR 功能对比分析

## 概述
本文档对比 Sequenzo 和 TraMineR 的功能，识别 Sequenzo 中缺失的功能。

**TraMineR 版本**: 2.2-13 (2025-12-14)  
**对比日期**: 2026-02-09

---

## 1. 序列数据定义和格式转换

### ✅ Sequenzo 已实现
- `SequenceData` - 定义状态序列对象
- 数据格式支持（宽格式、长格式）
- 缺失值处理

### ❌ Sequenzo 缺失的功能

#### 1.1 事件序列支持
- **`seqecreate()`** - 创建事件序列对象（Event Sequence）
- **`seqelist`** - 事件序列列表对象
- **`eseq`** - 单个事件序列对象
- **`is.eseq()` / `is.seqelist()`** - 检查事件序列对象类型
- **`seqelength()`** - 事件序列长度
- **`seqeweight()`** - 事件序列权重

#### 1.2 格式转换功能
- **`seqformat()`** - 更全面的格式转换功能：
  - STS ↔ SPELL 转换
  - STS ↔ TSE (Time-Stamped Event) 转换
  - STS ↔ SRS (Sequence of Repeated States) 转换
  - SPS (State-Period-Sequence) 格式支持
  - DSS (Distinct State Sequence) 格式转换
- **`seqdecomp()`** - 序列分解
- **`seqconc()`** - 序列连接
- **`seqsep()`** - 序列分离

#### 1.3 序列操作
- **`seqrecode()`** - 重新编码序列（合并状态）
- **`seqshift()`** - 序列移位
- **`seqasnum()`** - 序列转数字

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
- **`seqsubsn()`** - 子序列数量（更完整的实现）

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
  - OMslen, OMspell, OMspellNew, OMstran, OMslen
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
- **`OMloc`** - Localized Optimal Matching（可能已实现但需确认）
- **`OMtspell`** - Token-dependent spell OM（可能已实现但需确认）

#### 4.2 距离相关功能
- **`seqalign()`** - 序列对齐可视化（显示两个序列的对齐细节）
- **`seqfind()`** - 查找序列（在序列集中查找特定序列）
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
- **`seqrep()`** - 提取代表性序列集（支持多种标准：`'density'`, `'frequency'`）
- **`seqrf()`** - RF组中位数序列
- **`dissrep()`** - 从距离矩阵提取代表性对象
- **`dissrf()`** - RF组中位数
- **`disscenter()`** - 距离中心（虚拟中心或中位数）

#### 5.2 代表性序列可视化
- **`seqrplot()`** - 代表性序列图
- **`seqrfplot()`** - RF组中位数序列图

---

## 6. 事件序列分析

### ❌ Sequenzo 完全缺失的功能模块

#### 6.1 事件序列创建和处理
- **`seqecreate()`** - 创建事件序列对象
- **`seqeid()`** - 事件序列ID
- **`seqelength()`** - 事件序列长度
- **`seqeweight()`** - 事件序列权重
- **`seqeapplysub()`** - 应用子序列到事件序列

#### 6.2 事件子序列挖掘
- **`seqefsub()`** - 查找频繁事件子序列
- **`seqecmpgroup()`** - 比较组间事件子序列（识别最区分性的子序列）
- **`seqecontain()`** - 检查事件序列是否包含特定子序列
- **`seqeconstraint()`** - 事件序列约束（时间约束、计数方法等）

#### 6.3 事件序列转换
- **`seqetm()`** - 事件转换矩阵
- **`seqe2tse()`** - 事件序列转TSE格式

#### 6.4 事件序列可视化
- **`plot.subseqelist()`** - 事件子序列列表图
- **`plot.subseqelistchisq()`** - 事件子序列卡方检验图

---

## 7. 树分析

### ❌ Sequenzo 完全缺失的功能模块

#### 7.1 序列回归树
- **`seqtree()`** - 创建序列回归树（基于距离矩阵）
- **`seqtreedisplay()`** - 显示序列树
- **`seqtree2dot()`** - 序列树转GraphViz DOT格式

#### 7.2 距离树
- **`disstree()`** - 创建距离树
- **`disstreedisplay()`** - 显示距离树
- **`disstree2dot()`** - 距离树转DOT格式
- **`disstree2dotp()`** - 距离树转DOT格式（带参数）
- **`disstreeleaf()`** - 距离树叶节点
- **`disstree.get.rules()`** - 获取分类规则
- **`disstree.assign()`** - 分配规则索引

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
- **`seqdomassoc()`** - 域间状态关联（Piccarreta方法）
  - 支持Spearman相关
  - 返回p值

---

## 9. 距离矩阵分析

### ❌ Sequenzo 缺失的功能模块

#### 9.1 距离矩阵统计
- **`dissvar()`** - 距离矩阵的伪方差
- **`dissassoc()`** - 距离与因子的关联分析
- **`dissmfacw()`** - 多因子加权分析
- **`dissmergegroups()`** - 合并组以最小化分区质量损失

#### 9.2 距离矩阵可视化
- **`dissrep()`** - 代表性对象提取（已在上文列出）
- **`dissrf()`** - RF组中位数（已在上文列出）

---

## 10. 序列差异分析

### ❌ Sequenzo 缺失的功能

- **`seqdiff()`** - 序列差异分析（ANOVA-like分析）
  - 比较组间序列差异
  - 支持置换检验
  - 可视化功能 `plot.seqdiff()`

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
- **`seqdss()`** - 更完整的DSS（Distinct State Sequence）实现
- **`seqdur()`** - 更完整的持续时间计算（支持更多选项）

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

### ✅ Sequenzo 已实现
- 基本权重支持

### ❌ Sequenzo 缺失的功能

#### 14.1 加权统计
- 更全面的加权统计支持（在所有相关函数中）
- **`weighted.mean()`**, **`weighted.var()`**, **`weighted.fivenum()`** 等内部函数

#### 14.2 统计增强
- **`seqmeant()`** - 平均时间（支持 `serr` 参数显示标准误差）
- **`seqistatd()`** - 个体状态分布统计（支持 `prop` 参数计算比例）

---

## 总结

### 主要缺失的功能模块：

1. **事件序列分析** - 完全缺失
   - 事件序列对象创建和处理
   - 事件子序列挖掘
   - 事件序列可视化

2. **树分析** - 完全缺失
   - 序列回归树
   - 距离树
   - 树可视化

3. **代表性序列提取** - 部分缺失
   - `seqrep()` 和 `dissrep()` 功能
   - RF组中位数序列

4. **序列差异分析** - ✅ 已实现
   - `seqdiff()` ANOVA-like分析 (位于 `sequenzo/compare_differences/`)
   - `seqcompare()`, `seqLRT()`, `seqBIC()` 序列比较测试 (来自TraMineRextras)

5. **高级可视化** - 部分缺失
   - 平行坐标图
   - 熵指数图
   - 多域序列图

6. **综合指标函数** - 缺失
   - `seqindic()` 一次性计算多个指标

7. **格式转换** - 部分缺失
   - 更全面的格式转换（TSE, SRS等）
   - 事件序列格式支持

8. **工具函数** - 部分缺失
   - 序列检查函数
   - 序列查找函数
   - 序列对齐可视化

### 建议优先级：

**高优先级**：
1. 事件序列分析模块（如果用户需要）
2. 代表性序列提取（`seqrep()`, `dissrep()`）
3. 综合指标函数（`seqindic()`）
4. 序列差异分析（`seqdiff()`）

**中优先级**：
5. 树分析模块
6. 高级可视化（平行坐标图、熵图等）
7. 格式转换增强

**低优先级**：
8. 序列生成和模拟
9. 工具函数增强

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
