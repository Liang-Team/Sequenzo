# 在 Sequenzo 中实现 Helske et al. (2024) 的“从序列到变量”方法 — 实现需求与步骤

本文档**仅**基于 Helske et al. (2024) 文章正文与 Table 1 的描述，列出 sequenzo **尚未实现**的功能，并给出分阶段实现步骤。不加入文章未提及的算法或变量定义。

**参考文献**：Helske, S., Helske, J., & Chihaya, G. K. (2024). From Sequences to Variables: Rethinking the Relationship between Sequences and Outcomes. *Sociological Methodology*, 54(1), 27–51.  
本地全文：`developer/2024-helske.md`。

---

## 文章依据（避免幻觉）

以下每条需求均对应文章中的具体表述，实现时以文章为准。

| 需求 | 文章依据（2024-helske.md） |
|------|----------------------------|
| **Representativeness 公式** | 正文：*"we define the representativeness value of representative k (here, the medoid of cluster k) to sequence i as R_i^k = 1 - (distance of sequence i to representative k) / (maximum distance between two sequences). This leads to K continuous variables (which do not sum to 1)."* |
| **d_max 定义** | 同上："maximum distance between two sequences" → 即距离矩阵中任意两序列间距离的最大值。 |
| **Hard：聚类成员 → Dummies** | Table 1：Hard classification，Clustering method = Crisp (PAM)，Variable construction = Cluster membership，Variable type = Dummies。正文：*"one cluster is typically chosen as a reference, and the respective (dummy or probability) variable is omitted from the model."* |
| **Soft：FANNY → 成员概率** | Table 1：Soft classification，Fuzzy (FANNY)，Membership degree，Continuous。正文：*"using membership probabilities from fuzzy clustering as K continuous variables (which sum to 1 for each subject)"*，参考类省略。 |
| **FANNY 参数** | 正文：*"For FANNY, we fixed the membership exponent to 1.4, as larger values often led to complete fuzziness and convergence issues."* → 默认建议 1.4。 |
| **Pseudoclass 流程** | Table 1：Pseudoclass，Fuzzy (FANNY)，Multiple pseudoclass technique，Dummies。正文：*"individuals are randomly assigned to clusters multiple times on the basis of their membership probabilities, and for each sample estimate a model with a categorical membership variable the usual way. Finally, we combine the results across the models similarly to the multiple imputation technique (Rubin 2004)."* |

**不实现**：文章仅在补充材料中提及的 "gravity centers"（Batagelj 1988）等替代指标，本文档不要求实现。

---

## 从哪里开始（推荐阅读顺序）

1. **先看 1. 目标概览**：弄清四种方法分别是什么、当前 sequenzo 缺什么。
2. **再看 2. 实现原则与依赖关系**：理解“先 Representativeness、再 Hard 封装、再 Soft、最后 Pseudoclass”的顺序原因。
3. **按 3. 分阶段实现步骤** 从 **Phase 0** 做到 **Phase 1**，即可得到第一个可用的“代表性变量”API；之后按需做 Phase 2–5。
4. **实现时对照 4. 实现顺序小结** 和 **5. 验收与文档**；完成后可勾选 **6. 任务清单**。

---

## 1. 目标概览

### 1.1 四种方法（文章 Table 1）

| 方法 | 聚类算法 | 变量形式 | 用途 |
|------|----------|----------|------|
| **Hard classification** | 硬聚类 (PAM) | 聚类成员 → K 个虚拟变量 (dummies) | 传统做法 |
| **Soft classification** | 模糊聚类 (FANNY) | 成员概率 → K 个连续变量（每行和为 1） | 考虑归属不确定性 |
| **Pseudoclass** | FANNY + 多次抽样 | 每次为虚拟变量，多次拟合后按 Rubin 规则合并 | 考虑分类误差 |
| **Representativeness** | PAM 取 medoid | \(R_i^k = 1 - \frac{d(i,k)}{d_{\max}}\) → K 个连续变量 | 基于相似度的连续变量 |

### 1.2 Sequenzo 当前状态（简要）

- **Hard classification**：已有 PAM/KMedoids、CLARA、层次聚类，输出聚类标签；**缺少**：文章 Table 1 中的“变量构造”一步——将聚类成员转为 K（或 K−1，省略参考类）个 dummy 的封装。
- **Representativeness**：已有距离矩阵、medoid、`get_distance_matrix(..., refseq=...)`；**缺少**：文章中的公式 R_i^k = 1 − (distance to representative k) / (maximum distance between two sequences) 及返回 n×K 连续变量的 API。
- **Soft classification**：**未实现**。文章 Table 1 要求 Fuzzy (FANNY) 得到 membership degree（每行和为 1 的 K 个连续变量），sequenzo 目前无 FANNY。
- **Pseudoclass**：**未实现**。文章要求基于 FANNY 的成员概率多次抽样、每次用 categorical 变量拟合、再按 Rubin 规则合并，sequenzo 目前无此流程。

---

## 2. 实现原则与依赖关系

- **输入**：序列数据（SequenceData）+ 已有或可计算的**距离矩阵** `diss`（n×n）。
- **代表/中心**：由 PAM 或现有 KMedoids 得到 **K 个 medoid 索引**，作为“代表序列”；后续 Representativeness 与可选地 Soft/Pseudoclass 都基于同一套 medoid 或同一聚类方案，便于对比。
- **实现顺序建议**：先做 **Representativeness**（只用到现有 PAM + 距离），再做 **Hard 的 dummy 封装**，然后 **Soft（FANNY）**，最后 **Pseudoclass**（依赖 Soft 的成员概率）。

---

## 3. 分阶段实现步骤

### Phase 0：基础约定与工具（先做）

**目标**：统一“最大距离”与“从聚类标签到 dummy”的约定，供后续各方法复用。

#### Step 0.1 最大距离 `d_max`

- **定义**：\(d_{\max} = \max_{i,j} d(i,j)\)，即距离矩阵 `diss` 的上三角（或全矩阵）最大值。
- **要求**：
  - 在 `sequenzo/clustering/` 或 `sequenzo/utils/` 中提供**纯函数**，例如：
    - `def max_distance(diss: np.ndarray) -> float`
  - 输入：`diss` 为 n×n 对称矩阵（或 `scipy.dist` 的 condensed 形式时需先转成方阵再算）。
  - 输出：标量 `float`。
- **放置建议**：`sequenzo/clustering/seqs2vars_utils.py`（新建，见 Phase 4 文件结构）或放入现有 `sequenzo/clustering/utils/`。

#### Step 0.2 从聚类标签生成 dummy 矩阵

- **要求**：
  - 提供函数，例如：`cluster_labels_to_dummies(labels: np.ndarray, k: int = None, reference: int = 0) -> np.ndarray`
  - 输入：`labels` 为 0-based 或 1-based 的聚类编号，长度 n；`k` 为聚类数（若为 None 则用 `len(np.unique(labels))`）；`reference` 为参考类（省略的列），0 表示省略第一类。
  - 输出：n×(K-1) 的 0/1 矩阵（或 n×K 再在文档中说明哪一列是 reference）。
- **用途**：Hard classification 的“变量构造”步骤；Pseudoclass 每次抽样后也可复用。

---

### Phase 1：Representativeness 变量（优先实现）

**目标**：实现 \(R_i^k = 1 - \frac{d(i,\, \text{medoid}_k)}{d_{\max}}\)，返回 n×K 的连续变量矩阵（或 DataFrame），便于直接作为回归自变量。

#### Step 1.1 定义 API

- **函数名建议**：`representativeness_matrix(diss, medoid_indices, d_max=None)` 或 `compute_representativeness(diss, medoid_indices, max_dist=None)`。
- **参数**：
  - `diss`：n×n 距离矩阵（NumPy 或 DataFrame，若 DataFrame 需支持按行列索引取子块）。
  - `medoid_indices`：长度为 K 的整数数组/列表，每个元素为 medoid 在 `diss` 中的行/列索引（0-based）。
  - `d_max` / `max_dist`：标量；若为 `None`，则内部调用 Step 0.1 的 `max_distance(diss)`。
- **输出**：
  - 至少：**NumPy 数组**，形状 (n, K)，第 (i, k) 元素为 \(R_i^k\)。
  - 可选：**pandas DataFrame**，列名如 `R_1, R_2, ..., R_K` 或 `rep_medoid_1, ...`，索引与序列 ID 对齐（若调用方传入 `ids` 则设置 `df.index = ids`）。

#### Step 1.2 计算逻辑（严格按文章公式）

1. 若 `d_max is None`：`d_max = max_distance(diss)`（文章："maximum distance between two sequences"）。
2. 对每个 medoid 索引 `med_k`：取序列 i 到代表 k 的距离，即 `diss[i, med_k]`（或 `diss[med_k, :]` 的第 i 分量），形状 (n,)。
3. 文章公式：\(R_i^k = 1 - \frac{\text{distance of sequence } i \text{ to representative } k}{\text{maximum distance between two sequences}}\)。即 `R[:, k] = 1 - diss[:, med_k] / d_max`。若 `d_max == 0`，按约定处理（如全 1）。
4. 堆叠为 (n, K)。文章明确："This leads to K continuous variables (which do not sum to 1)"，故**不**省略任何列。

#### Step 1.3 与现有代码的衔接

- **Medoid 来源**：不强制在函数内做聚类；由调用方传入 `medoid_indices`。典型来源：
  - `KMedoids(diss, k=K, ...)` 的返回值是聚类标签，需再根据标签用现有 `disscenter`/类似逻辑得到每类的 medoid 索引；
  - 或 CLARA 的 `bestcluster['medoids']`（注意 CLARA 返回的可能是聚合后的索引，需映射回全样本索引）。
- **建议**：在文档/示例中给出“先 PAM/KMedoids → 取 medoid 索引 → 再调用 representativeness_matrix”的完整流程；若有现成的“从聚类结果取 medoid 列表”的辅助函数，可一并放在 Phase 0/1。

#### Step 1.4 文档与测试

- 在 docstring 中写明公式、参数含义、返回值形状；引用 Helske et al. (2024)。
- 单元测试：用小的 `diss` 和已知 medoid，手算 1−dist/d_max，对比输出。

---

### Phase 2：Hard classification 的“变量构造”封装

**目标**：让用户从“聚类结果”一步得到“可放入回归的 dummy 变量”，与文章 Table 1 的 Hard classification 一致。

#### Step 2.1 API

- **函数名建议**：`hard_classification_variables(labels, k=None, reference=0)` 或 `cluster_membership_to_dummies(...)`。
- **参数**：`labels`（1-based 或 0-based 聚类标签），`k`，`reference`（省略的类别）。
- **返回**：n×(K-1) 的 0/1 数组或 DataFrame（列名如 `C1, C2, ...` 表示相对 reference 的类别）。

#### Step 2.2 与 Step 0.2 的关系

- 若 Step 0.2 已实现 `cluster_labels_to_dummies`，本步可封装为：统一将 `labels` 转为 0-based、调用 Step 0.2、再可选地包成 DataFrame。避免重复实现。

---

### Phase 3：Soft classification（FANNY / 模糊聚类）

**目标**：基于距离矩阵得到“成员概率”矩阵 U（每行和为 1），作为 K 个连续自变量（或 K−1 个 + 参考类省略）。

#### Step 3.1 依赖与算法

- 文章使用 **FANNY**（Kaufman & Rousseeuw），基于**距离矩阵**的模糊 K-medoid 类算法，输出成员度（membership）\(u_{ik}\)。
- R 的 `cluster::fanny(..., diss = TRUE)` 接受距离矩阵；Python 需引入或实现等价算法。
- **可选方案**：
  - **A**：用 `sklearn_extra.cluster.FuzzyKMedoids`（若项目可接受依赖 `sklearn-contrib`），或其它接受 precomputed distance 的模糊聚类实现。
  - **B**：自实现一个基于 diss 的 FANNY 风格迭代（成员度更新 + medoid 更新），参考 R 的 FANNY 或文献。
- **建议**：先做方案 A（若有现成库），保证接口统一；否则再考虑 B。

#### Step 3.2 API

- **函数名建议**：`fanny_membership(diss, k, m=1.4, max_iter=100, tol=1e-6)` 或 `soft_clustering_membership(...)`。
- **参数**：`diss`（n×n），`k`，模糊系数 `m`（>1；**文章模拟中固定为 1.4**，见 2024-helske.md 正文），最大迭代次数与收敛容差。
- **返回**：
  - 成员概率矩阵 `U`，形状 (n, K)，每行和为 1；
  - 可选：medoid 索引（若算法内部会得到）。

#### Step 3.3 变量构造

- **Soft classification 变量**：即 U 的 K 列（或 K−1 列，省略参考类），作为连续变量直接用于回归。
- 可在同一模块提供：`soft_classification_variables(U, reference=0)` → 返回 (n, K) 或 (n, K−1) 的数组/DataFrame，列名如 `P1, P2, ...`。

#### Step 3.4 文档与测试

- 说明与 Helske 文中 “membership degree” 的对应关系；若使用固定参考类，在 docstring 中写明。
- 测试：用小型 diss 与已知结构，检查每行和是否为 1、数值是否合理。

---

### Phase 4：Pseudoclass

**目标**：基于 Soft 的成员概率 U，多次按概率抽样得到“伪类”标签，对每次伪类做回归并用量化规则合并（类似多重插补）。

#### Step 4.1 流程（与文章一致）

文章表述：*"individuals are randomly assigned to clusters multiple times on the basis of their membership probabilities, and for each sample estimate a model with a categorical membership variable the usual way. Finally, we combine the results across the models similarly to the multiple imputation technique (Rubin 2004)."*

1. 用 Phase 3 得到 U（n×K），每行和为 1。
2. 设重复次数 M（如 20）。
3. 对 m = 1,…,M：  
   - 对每个个体 i，按概率 `U[i, :]` 抽样得到类别 `c_i^(m)`；  
   - 用 `c_i^(m)` 构造 dummy（与 Hard 相同方式，复用 Step 0.2）；  
   - 用该 dummy 作为自变量拟合回归，得到 \(\hat\beta^{(m)}\) 及方差估计。  
4. 按 **Rubin 规则**（多重插补合并）合并：  
   - \(\bar\beta = \frac{1}{M}\sum_m \hat\beta^{(m)}\)；  
   - 组内方差 \(W\)、组间方差 \(B\)，总方差 \(T = W + (1+1/M)B\)；  
   - 标准误为 \(\sqrt{\text{diag}(T)}\)。

#### Step 4.2 API 设计（高层）

- **函数名建议**：`pseudoclass_regression(y, U, X_fixed, M=20, reference=0, fit_func=None)`。
- **参数**：
  - `y`：因变量（1维数组）。
  - `U`：成员概率矩阵 (n, K)。
  - `X_fixed`：其余自变量（截距、控制变量等），与 y 行对齐。
  - `M`：伪类重复次数。
  - `reference`：参考类。
  - `fit_func`：可选的拟合函数，签名为 `(y, X) -> (beta, se)` 或返回带 `.params` 和 `.bse` 的对象；默认可用 `statsmodels.OLS` 或 `sklearn.linear_model.LinearRegression` + 自举/解析标准误。
- **返回**：
  - 合并后的系数估计 `beta_combined`；
  - 合并后的标准误 `se_combined`；
  - 可选：每次的 \(\hat\beta^{(m)}\) 列表，便于诊断。

#### Step 4.3 依赖

- 依赖 Phase 0（dummy 构造）和 Phase 3（U）。
- 不强制依赖 Representativeness 或 Hard；Pseudoclass 仅用 U。

#### Step 4.4 文档与测试

- 在 docstring 中引用 Bandeen-Roche et al.、Lanza et al. 及 Helske 的 pseudoclass 描述。
- 测试：用模拟 U 和简单线性模型，检查合并后系数与理论期望是否接近（例如已知数据生成过程时）。

---

### Phase 5：统一入口与模块结构（可选但推荐）

**目标**：提供单一入口函数和清晰模块划分，方便用户按“序列 + 距离 → 四种变量”使用。

#### Step 5.1 模块与文件建议

- **新建**：`sequenzo/clustering/sequences_to_variables.py`（或 `sequenzo/seqs2vars/` 子包）。
  - 内含：
    - `max_distance(diss)`
    - `cluster_labels_to_dummies(labels, k=None, reference=0)`
    - `representativeness_matrix(diss, medoid_indices, d_max=None)`（Phase 1）
    - `hard_classification_variables(labels, k=None, reference=0)`（Phase 2，可调用 Step 0.2）
    - `fanny_membership(diss, k, ...)` 与 `soft_classification_variables(U, reference=0)`（Phase 3）
    - `pseudoclass_regression(y, U, X_fixed, M=20, ...)`（Phase 4）
- **辅助模块**（若单独放）：`sequenzo/clustering/seqs2vars_utils.py`，仅放 `max_distance`、`cluster_labels_to_dummies` 等纯工具，供 `sequences_to_variables` 和未来 CLARA 扩展（如 representativeness 模式）共用。

#### Step 5.2 统一入口（可选）

- 例如：`sequences_to_variables(seqdata, diss, method, k, **kwargs)`。
  - `method in ("hard", "soft", "pseudoclass", "representativeness")`。
  - 内部根据 method 调用 KMedoids/PAM 或 FANNY，再调用对应变量构造函数；若 `method="representativeness"` 则只做 PAM 取 medoid + `representativeness_matrix`。
  - 返回结构可统一为：`dict` 含 `"X"`（变量矩阵/DataFrame）、`"medoids"`（若适用）、`"labels"`（若适用）、`"U"`（若 soft/pseudoclass）等，便于下游回归与比较。

#### Step 5.3 与现有包结构的衔接

- **KMedoids**：保持现有 `sequenzo.clustering.KMedoids.KMedoids` 不变；新模块只“调用”它得到标签或 medoid，不修改其返回值约定。
- **CLARA**：若将来在 CLARA 中支持 `method="representativeness"`，可在内部对全样本用 `representativeness_matrix(diss_full, medoids_from_clara, d_max)` 得到 R 矩阵，与本文档 Phase 1 的 API 一致。
- **get_distance_matrix**：继续作为“距离矩阵 + refseq”的唯一入口；Representativeness 的 `diss` 可由用户用 `get_distance_matrix(seqdata, ...)` 得到，再传入 `representativeness_matrix`。

---

## 4. 实现顺序小结（按依赖关系）

1. **Phase 0**：`max_distance`、`cluster_labels_to_dummies`（及可选：从聚类标签得到 medoid 索引的辅助函数）。
2. **Phase 1**：`representativeness_matrix(diss, medoid_indices, d_max=None)`，并写清“PAM → medoid → R”的示例。
3. **Phase 2**：`hard_classification_variables(...)`（可薄封装 Step 0.2）。
4. **Phase 3**：FANNY/模糊聚类 → `U`；`soft_classification_variables(U, reference=0)`。
5. **Phase 4**：`pseudoclass_regression(y, U, X_fixed, M=20, ...)`。
6. **Phase 5**：整理到 `sequences_to_variables` 模块与统一入口（可选）。

---

## 5. 验收与文档

- **单元测试**：每个 Phase 对应至少一个测试文件或测试函数，覆盖正常输入与边界（如 k=2、d_max=0）。
- **示例脚本/笔记本**：在 `developer/` 或 `examples/` 中提供一个最小示例：用同一份 `seqdata` 与 `diss`，依次生成四种变量并跑简单回归，与文章结论（如 Representativeness 在 similarity-based 设定下更优）可对照说明。
- **文档**：在 sequenzo 用户文档中增加“从序列到变量（Helske 方法）”小节，列出四种方法、适用场景、API 与参考文献。

---

## 6. 参考文献（文中直接引用）

- Helske et al. (2024). From Sequences to Variables. *Sociological Methodology* 54(1), 27–51.
- Kaufman, L., & Rousseeuw, P. J. (2009). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
- Bandeen-Roche et al. (1997). Latent Variable Regression for Multiple Discrete Outcomes. *JASA* 92(440), 1375–1386.
- Rubin, D. B. (2004). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

---

## 7. 任务清单（实现时可勾选）

| 阶段 | 任务 | 状态 |
|------|------|------|
| **Phase 0** | 实现 `max_distance(diss)` | ☐ |
| | 实现 `cluster_labels_to_dummies(labels, k, reference)` | ☐ |
| | （可选）从聚类标签/ KMedoids 结果得到 medoid 索引的辅助函数 | ☐ |
| **Phase 1** | 实现 `representativeness_matrix(diss, medoid_indices, d_max=None)` | ☐ |
| | 单元测试 + 文档/示例（PAM → medoid → R） | ☐ |
| **Phase 2** | 实现 `hard_classification_variables(labels, k, reference)` | ☐ |
| | 与 Step 0.2 复用，避免重复 | ☐ |
| **Phase 3** | 引入或实现 FANNY/模糊聚类 → 成员概率 U | ☐ |
| | 实现 `soft_classification_variables(U, reference)` | ☐ |
| **Phase 4** | 实现 `pseudoclass_regression(y, U, X_fixed, M, ...)` | ☐ |
| | Rubin 规则合并系数与标准误 | ☐ |
| **Phase 5** | 新建 `sequences_to_variables` 模块并集中上述 API | ☐ |
| | （可选）统一入口 `sequences_to_variables(seqdata, diss, method, k)` | ☐ |
| **验收** | 各 Phase 单元测试通过 | ☐ |
| | 示例脚本/笔记本：四种变量 + 简单回归 | ☐ |
| | 用户文档“从序列到变量”小节 | ☐ |

---

## 8. 仅实现文章描述且 sequenzo 未实现的功能（核对用）

- **Representativeness**：文章给出唯一公式与“K continuous variables (do not sum to 1)”；sequenzo 无此 API → 需实现。
- **Hard 的变量构造**：文章 Table 1 为 cluster membership → Dummies，参考类省略；sequenzo 有聚类标签、无“标签→dummy 矩阵”封装 → 可做封装。
- **Soft**：文章 Table 1 为 FANNY → membership degree（K 连续，和为 1），参考类省略；sequenzo 无 FANNY → 需实现。
- **Pseudoclass**：文章为按 U 多次抽样→每次 dummy 回归→Rubin 合并；sequenzo 无 → 需实现。
- **不实现**：gravity centers、AMPs/AMEs（文章用于结果展示，非变量构造）、其他文章未明确描述的算法。

---

*文档版本：1.1 | 严格依据 Helske et al. (2024) 正文与 Table 1，仅列 sequenzo 未实现部分*
