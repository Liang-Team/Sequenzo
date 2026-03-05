# 聚类功能对标：WeightedCluster (R) vs Sequenzo (Python)

本文档对照 **WeightedCluster**（R 包，`developer/WeightedCluster-master`）与 **Sequenzo** 的 **clustering** 相关功能，便于查漏补缺与对标。  
**说明**：Sequenzo 的「大样本 CLARA」在 `sequenzo.big_data.clara`，树分析在 `sequenzo.tree_analysis`，此处只对标 **clustering** 模块及与聚类直接相关的“序列→变量”能力。

---

## 一、总览表（按功能域）

| 功能域 | 功能点 | WeightedCluster | Sequenzo (clustering) | 备注 |
|--------|--------|-----------------|------------------------|------|
| **1. 层次聚类** | 层次聚类 + 多种方法 | ✅ wcCmpCluster / as.clustrange(hclust) | ✅ Cluster (fastcluster) | 见 二.1 |
| **2. 聚类质量** | 多 k 质量指标 (PBC, ASW, CH, R2, HC…) | ✅ wcClusterQuality, as.clustrange | ✅ ClusterQuality | 见 二.2 |
| **2. 聚类质量** | 个体轮廓值 (ASW per observation) | ✅ wcSilhouetteObs | ❌ 未实现 | 见 二.2 |
| **3. 划分与结果** | 从层次/划分得到多 k 成员 + 质量 | ✅ as.clustrange, wcKMedRange | ✅ ClusterResults + ClusterQuality | 见 二.3 |
| **4. K-Medoids / PAM** | 加权 PAM，支持初始划分 | ✅ wcKMedoids | ✅ KMedoids | 见 二.4 |
| **4. K-Medoids / PAM** | 多 k 的 PAM 一次跑 + 质量 | ✅ wcKMedRange | ⚠️ 需自己循环 KMedoids + ClusterQuality | 见 二.4 |
| **4. K-Medoids / PAM** | 求 medoid 索引（按组/权重） | ✅ disscenter (内部) | ✅ disscentertrim | 见 二.4 |
| **5. 大样本 / CLARA** | 聚合重复个案 | ✅ wcAggregateCases | ⚠️ 在 big_data.clara 有 aggregatecases | 见 二.5 |
| **5. 大样本 / CLARA** | CLARA 式抽样 + 多 k + 选最优 k | ✅ seqclararange | ⚠️ 有 big_data.clara，接口不同 | 见 二.5 |
| **5. 大样本 / CLARA** | CLARA 方法：crisp / fuzzy / representativeness / noise | ✅ method="crisp"等 | ❌ clustering 内无 seqclararange 式四方法 | 见 二.5 |
| **6. 序列→变量 (Helske)** | Representativeness R_i^k | ✅ representativeness (seqclararange) | ✅ representativeness_matrix | 见 二.6 |
| **6. 序列→变量 (Helske)** | Hard：标签 → K−1 dummy | ❌ 无封装 | ✅ hard_classification_variables | 见 二.6 |
| **6. 序列→变量 (Helske)** | Soft：FANNY → K−1 连续变量 | ❌ 无封装 | ✅ soft_classification_variables | 见 二.6 |
| **6. 序列→变量 (Helske)** | Pseudoclass（多次抽样 + Rubin 合并） | ❌ 无 | ✅ pseudoclass_regression | 见 二.6 |
| **7. 模糊聚类** | FANNY 成员度 | ✅ fanny + fuzzy 方法 | ✅ fanny_membership | 见 二.7 |
| **7. 模糊聚类** | FCMdd 迭代（基于距离的模糊 C 均值） | ✅ wfcmdd | ❌ 无 | 见 二.7 |
| **7. 模糊聚类** | 模糊聚类可视化 (seqplot 按成员加权) | ✅ fuzzyseqplot | ❌ 无 | 见 二.7 |
| **8. 验证与关联** | 聚类质量 bootstrap（选 k 稳定性） | ✅ bootclustrange | ❌ 无 | 见 二.8 |
| **8. 验证与关联** | 聚类与协变量关联 (dissmfacw + BIC) | ✅ clustassoc | ❌ 无 | 见 二.8 |
| **8. 验证与关联** | RARCAT（typology + 回归的稳健性） | ✅ rarcat | ❌ 无 | 见 二.8 |
| **9. 属性/树聚类** | 基于协变量的 disstree / seqtree | ✅ propclustering, seqpropclust | ✅ tree_analysis (disstree, seqtree) | 不在 clustering 内 | 见 二.9 |
| **10. 其他** | Davies-Bouldin 等 | ✅ davies_bouldin_internal | ❌ 未单独暴露 | 见 二.10 |

---

## 二、分项说明

### 1. 层次聚类

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **入口** | `wcCmpCluster(diss, maxcluster, method="all")` 可一次跑多种方法；或 `hclust()` + `as.clustrange(hc, diss, ncluster=…)` | `Cluster(matrix, clustering_method="ward_d")` 等，单方法 |
| **方法** | ward.D, ward.D2, single, complete, average, mcquitty, median, centroid, pam, diana, beta.flexible | 通过 fastcluster：single, complete, average, weighted, ward, ward_d2, centroid, median 等 |
| **权重** | 支持 `weights`（hclust 的 members） | 支持 `weights`（用于质量与结果，不用于 linkage 计算） |
| **多方法一次比较** | ✅ 一次得到多方法、多 k 的质量表 | ❌ 需对每种方法分别建 Cluster + ClusterQuality |

**结论**：层次聚类与单方法多 k 质量两边都有；**多方法一次比较**仅 WeightedCluster 有。

---

### 2. 聚类质量指标

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **指标** | PBC, HG, HGSD, ASW, ASWw, CH, R2, CHsq, R2sq, HC | 与 R 包一致（C++ 对齐 WeightedCluster） |
| **单次划分** | `wcClusterQuality(diss, clustering, weights)` | 通过 `ClusterQuality` 对多 k 计算，内部调 C++ |
| **个体轮廓** | `wcSilhouetteObs(diss, clustering, weights, measure="ASW"\|"ASWw")` 返回每观测的 ASW | ❌ 未提供每观测轮廓值 |

**结论**：整体质量指标**已对标**；**个体级 ASW** 仅 WeightedCluster 有。

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
| **多 k 一次** | `wcKMedRange(diss, kvals, weights)` 直接得到多 k 的划分 + 质量表 | 需对每个 k 调用 `KMedoids`，再对每个划分算质量（或自己封装循环） |
| **Medoid 索引** | 内部 `disscenter(diss, group=clustering, medoids.index=…)` | `disscentertrim(diss, group=..., medoids_index=...)`（C++） |

**结论**：单次 PAM **已对标**；**多 k 的 KMedRange 式一键接口** 仅 WeightedCluster 有。

---

### 5. 大样本 / CLARA 与聚合

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **聚合重复个案** | `wcAggregateCases(seqdata, weights)` → aggIndex, aggWeights, disaggIndex | `sequenzo.big_data.clara.utils.aggregatecases` 有等价逻辑，不在 clustering 下 |
| **CLARA 式流程** | `seqclararange(seqdata, kvals, method="crisp"\|"fuzzy"\|"representativeness"\|"noise", max.dist=…)`：抽样 → 距离 → 聚类 → 全样本到 medoid 距离 → 质量/代表度 | `sequenzo.big_data.clara.clara`：接口与流程不同，不直接对应 seqclararange 的四种 method |
| **Representativeness 输出** | `method="representativeness"` 时返回 `1 - diss/max.dist` 的 n×K 矩阵 | clustering 内用 `representativeness_matrix(diss, medoid_indices, d_max)`，不依赖 CLARA |

**结论**：**聚合** 两边都有（Sequenzo 在 big_data）；**seqclararange 式的四方法 CLARA（crisp/fuzzy/representativeness/noise）** 仅 WeightedCluster 有；Sequenzo 的 representativeness 以「给定 medoid + 距离矩阵」的独立函数提供。

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
| **FANNY 成员度** | `fanny(diss, k, diss=TRUE, memb.exp=m)`（R cluster） | `fanny_membership(diss, k, m=1.4)`（基于 PAM medoid + 距离公式） |
| **FCMdd 迭代** | `wfcmdd(diss, memb, weights, method="FCMdd", m=…)` 迭代优化 | ❌ 无 |
| **模糊可视化** | `fuzzyseqplot(seqdata, group=fanny_object, ...)` 按成员加权画序列 | ❌ 无 |

**结论**：**成员度** 两边都有（实现方式不同）；**FCMdd** 与 **模糊序列图** 仅 WeightedCluster 有。

---

### 8. 验证与关联

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **Bootstrap 选 k** | `bootclustrange(object, seqdata, seqdist.args, R=100, ...)` 对 seqclararange 等做 bootstrap 质量 | ❌ 无 |
| **聚类与协变量** | `clustassoc(clustrange, diss, covar, weights)`：dissmfacw 看“未被聚类解释的关联”+ 用聚类做回归的 BIC | ❌ 无 |
| **RARCAT** | `rarcat(formula, data, diss, ncluster, ...)`：bootstrap 建 typology + 回归，多水平合并 | ❌ 无 |

**结论**：**bootstrap 验证、clustassoc、RARCAT** 仅 WeightedCluster 有。

---

### 9. 属性/树聚类（非 clustering 模块）

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **基于距离 + 协变量的树** | `seqtree`, `disstree`（TraMineR 等），`propclustering` 中的 `dtprune`, `clusterSplitSchedule` | `tree_analysis.build_sequence_tree` / `build_distance_tree` 等 |

**结论**：两边都有「用协变量切分距离」的树，Sequenzo 在 `tree_analysis`，不在 `clustering`。

---

### 10. 其他

| 项目 | WeightedCluster | Sequenzo |
|------|-----------------|----------|
| **Davies-Bouldin** | 内部 `davies_bouldin_internal`，用于 fuzzy/crisp 质量 | 未单独暴露为 API |

---

## 三、简要结论（对标用）

- **WeightedCluster 有而 Sequenzo clustering 暂无的**  
  - 多方法一次比较（wcCmpCluster）；  
  - 个体轮廓值（wcSilhouetteObs）；  
  - 多 k 一键 PAM（wcKMedRange）；  
  - seqclararange 式四方法 CLARA（crisp/fuzzy/representativeness/noise）；  
  - 聚合在包内（wcAggregateCases）；  
  - FCMdd（wfcmdd）、模糊序列图（fuzzyseqplot）；  
  - bootstrap 选 k（bootclustrange）、clustassoc、RARCAT。

- **Sequenzo clustering 有而 WeightedCluster 暂无的**  
  - Helske 式「回归用变量」封装：hard_classification_variables、soft_classification_variables；  
  - pseudoclass_regression（多次抽样 + Rubin 合并）；  
  - max_distance、cluster_labels_to_dummies 等序列→变量辅助。

- **两边都有的（已对标或等价）**  
  - 层次聚类 + 多 k 质量（PBC, ASW, CH, R2, HC 等）；  
  - 加权 PAM（KMedoids）；  
  - medoid 计算（disscenter / disscentertrim）；  
  - Representativeness 公式 R_i^k；  
  - FANNY 成员度（或等价的基于距离的成员度）。

如需在 Sequenzo 中「对齐」WeightedCluster，可优先考虑：个体 ASW、wcKMedRange 式多 k 接口、以及（若要做大样本）seqclararange 式 CLARA 与 bootstrap/验证（bootclustrange、clustassoc）的对应实现或文档说明。
