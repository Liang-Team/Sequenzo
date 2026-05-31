# 聚类功能对标：WeightedCluster (R) vs Sequenzo (Python)

本文档对照 **WeightedCluster**（R 包，`developer/WeightedCluster-master`）与 **Sequenzo** 的 **clustering** 相关功能，便于查漏补缺。
**说明**：Sequenzo 的「大样本 CLARA」在 `sequenzo.big_data.clara`，树分析在 `sequenzo.discrepancy_analysis`，此处只比较 **clustering** 模块及与聚类直接相关的“序列→变量”能力。文中的 R 对齐结论均以当前记录测试与已列明边界为准，不表示所有输入、随机初值或退化情形都逐点等同。

---

## 一、总览表（按功能域）

| 功能域 | 功能点 | WeightedCluster | Sequenzo (clustering) | 备注 |
|--------|--------|-----------------|------------------------|------|
| **1. 层次聚类** | 层次聚类 + 多种方法 | ✅ wcCmpCluster / as.clustrange(hclust) | ✅ Cluster (fastcluster) | 见 二.1 |
| **2. 聚类质量** | 多 k 质量指标 (PBC, ASW, CH, R2, HC…) | ✅ wcClusterQuality, as.clustrange | ✅ ClusterQuality | 见 二.2 |
| **2. 聚类质量** | 个体轮廓值 (ASW per observation) | ✅ wcSilhouetteObs | ✅ observation_silhouette | 见 二.2 |
| **3. 划分与结果** | 从层次/划分得到多 k 成员 + 质量 | ✅ as.clustrange, wcKMedRange | ✅ ClusterResults + ClusterQuality | 见 二.3 |
| **4. K-Medoids / PAM** | 加权 PAM，支持初始划分 | ✅ wcKMedoids | ✅ KMedoids | 见 二.4 |
| **4. K-Medoids / PAM** | 多 k 的 PAM 一次跑 + 质量 | ✅ wcKMedRange | ✅ k_medoids_range | 见 二.4 |
| **4. K-Medoids / PAM** | 求 medoid 索引（按组/权重） | ✅ disscenter (内部) | ✅ disscentertrim | 见 二.4 |
| **5. 大样本 / CLARA** | 聚合重复个案 | ✅ wcAggregateCases | ✅ aggregate_cases | 见 二.5 |
| **5. 大样本 / CLARA** | CLARA 式抽样 + 多 k + 选最优 k | ✅ seqclararange | ✅ seqclara_range / clara | 见 二.5 |
| **5. 大样本 / CLARA** | CLARA 方法：crisp / fuzzy / representativeness / noise | ✅ method="crisp"等 | ✅ method="crisp"等 | 见 二.5 |
| **6. 序列→变量 (Helske)** | Representativeness R_i^k | ✅ representativeness (seqclararange) | ✅ representativeness_matrix | 见 二.6 |
| **6. 序列→变量 (Helske)** | Hard：标签 → K−1 dummy | ❌ 无封装 | ✅ hard_classification_variables | 见 二.6 |
| **6. 序列→变量 (Helske)** | Soft：FANNY → K−1 连续变量 | ❌ 无封装 | ✅ soft_classification_variables | 见 二.6 |
| **6. 序列→变量 (Helske)** | Pseudoclass（多次抽样 + Rubin 合并） | ❌ 无 | ✅ pseudoclass_regression | 见 二.6 |
| **7. 模糊聚类** | FANNY 成员度 | ✅ fanny + fuzzy 方法 | ✅ fanny_membership | 见 二.7 |
| **7. 模糊聚类** | FCMdd 迭代（基于距离的模糊 C 均值） | ✅ wfcmdd | ✅ wfcmdd | 见 二.7 |
| **7. 模糊聚类** | 模糊聚类可视化 (seqplot 按成员加权) | ✅ fuzzyseqplot | ✅ fuzzy_sequence_plot | 见 二.7 |
| **8. 验证与关联** | 聚类质量 bootstrap（选 k 稳定性） | ✅ bootclustrange | ✅ boot_cluster_range | 见 二.8 |
| **8. 验证与关联** | 聚类与协变量关联 (dissmfacw + BIC) | ✅ clustassoc | ✅ cluster_association | 见 二.8 |
| **8. 验证与关联** | RARCAT（typology + 回归的稳健性） | ✅ rarcat | ✅ rarcat | 见 二.8 |
| **9. 属性/树聚类** | 基于协变量的 disstree / seqtree | ✅ propclustering, seqpropclust | ✅ discrepancy_analysis (disstree, seqtree) | 不在 clustering 内 | 见 二.9 |
| **10. 其他** | Davies-Bouldin 等 | ✅ davies_bouldin_internal | ❌ 未单独暴露 | 见 二.10 |

---

## 二、分项说明

### 1. 层次聚类

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **入口** | `wcCmpCluster(diss, maxcluster, method="all")` 可一次跑多种方法；或 `hclust()` + `as.clustrange(hc, diss, ncluster=…)` | `Cluster(matrix, clustering_method="ward_d")` 等，单方法 |
| **方法** | ward.D, ward.D2, single, complete, average, mcquitty, median, centroid, pam, diana, beta.flexible | 通过 fastcluster：single, complete, average, weighted, ward, ward_d2, centroid, median 等 |
| **权重** | 支持 `weights`（hclust 的 members） | 支持 `weights`（`get_weighted_diss` + linkage；与 R `hclust(..., members=weights)` 一致） |
| **多方法一次比较** | ✅ 一次得到多方法、多 k 的质量表 | ✅ compare_cluster_methods | 见 二.1 |

**结论**：层次聚类、单方法多 k 质量与多方法一次比较均有对应实现；`diana` / `beta.flexible` 在 Python 内实现，当前仅在无权重情形使用，带权时与 R 一样排除。

---

### 2. 聚类质量指标

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **指标** | PBC, HG, HGSD, ASW, ASWw, CH, R2, CHsq, R2sq, HC | C++ 实现；在记录测试范围内与 WeightedCluster 参考结果对齐 |
| **单次划分** | `wcClusterQuality(diss, clustering, weights)` | 通过 `ClusterQuality` 对多 k 计算，内部调 C++ |
| **个体轮廓** | `wcSilhouetteObs(diss, clustering, weights, measure="ASW"\|"ASWw")` 返回每观测的 ASW | `observation_silhouette(diss, clustering, weights, measure="ASW"\|"ASWw")`（C++ 实现；记录测试范围内与 WeightedCluster 参考结果对齐） |

**结论**：整体质量指标与个体级 ASW 均有对应实现，并在记录测试范围内与 WeightedCluster 参考结果对齐。

---

### 3. 划分与结果输出

| 项目 | WeightedCluster | Sequenzo |
|------|----------------|----------|
| **从层次得成员** | `cutree(hc, k)`，再 `as.clustrange(..., diss)` 得质量 | `ClusterResults(cluster).get_cluster_memberships(num_clusters)` |
| **聚类分布/加权统计** | 需自己用 `table` 等 | `ClusterResults(...).get_cluster_distribution(..., weighted=True)` |
| **树状图** | 依赖 TraMineR/外部 | `Cluster.plot_dendrogram()` |

**结论**：两边都能从层次得到成员与分布；Sequenzo 在「分布 + 加权」上封装更直接。

---

### 4. K-Medoids / PAM

| 项目 | WeightedCluster | Sequenzo |
|------|----------------|----------|
| **加权 PAM** | `wcKMedoids(diss, k, weights, initialclust, method="PAMonce", cluster.only=…)` | `KMedoids(diss, k, weights, initialclust, method='PAMonce', cluster_only=…)` |
| **多 k 一次** | `wcKMedRange(diss, kvals, weights)` 直接得到多 k 的划分 + 质量表 | `k_medoids_range(diss, kvals, weights)`；纯 Python 实现，`random_state` 控制 NumPy 初值 |
| **Medoid 索引** | 内部 `disscenter(diss, group=clustering, medoids.index=…)` | `disscentertrim(diss, group=..., medoids_index=...)`（C++） |

**结论**：单次 PAM 与多 k 一键 PAM 均有对应实现，并在记录测试范围内与 WeightedCluster 参考结果对齐。

---

### 5. 大样本 / CLARA 与聚合

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **聚合重复个案** | `wcAggregateCases(seqdata, weights)` → aggIndex, aggWeights, disaggIndex | `aggregate_cases(seqdata, weights)`；`agg_index` / `disagg_index` 为 **1-based**（沿用 R 的索引约定） |
| **CLARA 式流程** | `seqclararange(seqdata, kvals, method="crisp"\|"fuzzy"\|"representativeness"\|"noise", max.dist=…)` | `seqclara_range(...)` / `clara(...)`（`sequenzo.big_data.clara`） |
| **Representativeness 输出** | `method="representativeness"` 时返回 `1 - diss/max.dist` 的 n×K 矩阵 | `method="representativeness"` 时在 CLARA 结果中返回代表度矩阵 |

**结论**：**聚合** 与 **seqclararange 式四方法 CLARA** 两边均有对应实现（CLARA 在 `big_data.clara`）。

---

### 6. 序列→变量（Helske 表 1）

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **Representativeness** | ✅ `seqclararange(..., method="representativeness")`，公式 R = 1 - diss/max.dist | ✅ `representativeness_matrix(diss, medoid_indices, d_max)` |
| **Hard：标签 → dummy** | ❌ 只给聚类标签，无「K−1 个 dummy」封装 | ✅ `hard_classification_variables(labels, k, reference=0)` |
| **Soft：成员度 → 回归用变量** | ❌ 有 membership，无「省略参考类后的 K−1 连续变量」 | ✅ `soft_classification_variables(U, reference=0)` |
| **Pseudoclass** | ❌ 无「多次抽样 + Rubin 合并」 | ✅ `pseudoclass_regression(y, U, X_fixed, M=20, reference=0)` |
| **辅助** | 需用户自己算 max.dist、造 dummy | ✅ `max_distance(diss)`, `cluster_labels_to_dummies(...)` |

**结论**：Representativeness 两边都有；**Hard/Soft 的回归用变量封装** 和 **Pseudoclass** 仅 Sequenzo 在 clustering 中实现，与 Helske 表 1 对齐。

---

### 7. 模糊聚类

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **FANNY 成员度** | `fanny(diss, k, diss=TRUE, memb.exp=m)`（R cluster） | `fanny_membership(diss, k, m=1.4)`（R-derived 距离矩阵 FANNY；已测试的多簇非退化情形与 R 目标对齐。`k=1` 为 Sequenzo 确定性快捷路径；全零距离等无法形成多簇成员度的退化输入会被拒绝。PAM-medoid 距离公式仅用于 `medoid_membership_approximation` 快速启发式） |
| **FCMdd 迭代** | `wfcmdd(diss, memb, weights, method="FCMdd", m=…)` 迭代优化 | `wfcmdd(diss, memb, weights, method="FCMdd", m=…)` |
| **模糊可视化** | `fuzzyseqplot(seqdata, group=fanny_object, ...)` 按成员加权画序列 | `fuzzy_sequence_plot(seqdata, membership, ...)` 按成员加权画序列 |

**结论**：**成员度、FCMdd 与模糊序列图** 两边均有对应实现（接口与绘图栈不同）。

---

### 8. 验证与关联

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **Bootstrap 选 k** | `bootclustrange(object, seqdata, seqdist.args, R=100, ...)` 对 seqclararange 等做 bootstrap 质量 | `boot_cluster_range(clustering, distance_builder, n_boot=…, sample_size=…)` |
| **聚类与协变量** | `clustassoc(clustrange, diss, covar, weights)`：dissmfacw 看“未被聚类解释的关联”+ 用聚类做回归的 BIC | `cluster_association(clustrange, diss, covar, weights)` |
| **RARCAT** | `rarcat(formula, data, diss, ncluster, ...)`：bootstrap 建 typology + 回归，多水平合并 | `rarcat(formula, data, diss, ncluster, ...)` |

**结论**：**bootstrap 验证、clustassoc、RARCAT** 两边均有对应实现（入口参数与依赖对象不同）。

---

### 9. 属性/树聚类（非 clustering 模块）

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **基于距离 + 协变量的树** | `seqtree`, `disstree`（TraMineR 等），`propclustering` 中的 `dtprune`, `clusterSplitSchedule` | `discrepancy_analysis.build_sequence_tree` / `build_distance_tree` 等 |

**结论**：两边都有「用协变量切分距离」的树，Sequenzo 在 `discrepancy_analysis`，不在 `clustering`。

---

### 10. 其他

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **Davies-Bouldin** | 内部 `davies_bouldin_internal`，用于 fuzzy/crisp 质量 | 未单独暴露为 API |

---

## 三、简要结论（比较用）

- **WeightedCluster 有而 Sequenzo clustering 暂无的**  
  - 暂未列出未覆盖项；`diana` / `beta.flexible` 已在 `compare_cluster_methods` 中提供 Python 实现，带权时同样不可用。

- **Sequenzo clustering 有而 WeightedCluster 暂无的**  
  - Helske 式「回归用变量」封装：hard_classification_variables、soft_classification_variables；  
  - pseudoclass_regression（多次抽样 + Rubin 合并）；  
  - max_distance、cluster_labels_to_dummies 等序列→变量辅助。

- **两边都有对应实现的**
  - 层次聚类 + 多 k 质量（PBC, ASW, CH, R2, HC 等）；  
  - 多方法一次比较（wcCmpCluster / compare_cluster_methods）；  
  - 多 k 一键 PAM（wcKMedRange / k_medoids_range）；  
  - 聚合（wcAggregateCases / aggregate_cases）；  
  - seqclararange 式四方法 CLARA（`sequenzo.big_data.clara`）；  
  - 个体轮廓值（wcSilhouetteObs / observation_silhouette）；  
  - 加权 PAM（KMedoids）；  
  - medoid 计算（disscenter / disscentertrim）；  
  - Representativeness 公式 R_i^k；  
  - FANNY 成员度（R-derived 距离矩阵实现；R parity 限于测试覆盖的多簇非退化情形）；
  - FCMdd（wfcmdd）、模糊序列图（fuzzyseqplot / fuzzy_sequence_plot）；  
  - bootstrap 选 k（bootclustrange / boot_cluster_range）、clustassoc / cluster_association、RARCAT。

**使用注意（R 对齐边界）**
- `aggregate_cases`：`agg_index`、`disagg_index` 为 **1-based**；`agg_weights`、`disagg_weights` 在记录测试范围内与 R 参考结果一致。
- `KMedoids` 返回 **1-based medoid 行号**（同 `wcKMedoids`）；`medoid_indices_from_kmedoids_result` / `cluster_labels_from_kmedoids_result` 提供 0-based medoid 行号与 0..K-1 簇标签。  
- `k_medoids_range`：纯 Python 编排；`random_state` 控制 NumPy 初值，随机划分未必与 R 逐点相同；划分质量仅在记录的同一划分参考测试中与 WeightedCluster C++ 结果对齐。
- `compare_cluster_methods`：`diana` / `beta.flexible` 为纯 Python 实现；`methods="all"` 且带权时自动排除二者；显式指定且带权会报错，行为与 R 的带权限制一致。
- 包级同名导出（例如 `sequenzo.clustering.k_medoids_range` 和 `sequenzo.dissimilarity_measures.get_distance_matrix`）采用 callable submodule 契约：`from package import name` 后可直接调用，普通 dotted import 后也保留 `package.name.name` 的子模块属性链与函数签名。原始 `importlib.import_module("...same_named_module")` 返回标准模块对象，不承诺该模块本身可直接调用；需要可调用导出时使用 dotted import 或 from-import。该契约由 `tests/clustering/test_import_contracts.py` 固定；严格依赖 `inspect.isfunction()` 的外部代码应改为使用 `callable()`。
