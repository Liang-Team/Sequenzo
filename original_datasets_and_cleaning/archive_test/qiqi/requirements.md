# Sequenzo HMM 功能实现需求文档

## 📋 当前情况概述

### 1. 现有资源

#### seqHMM (R包) - 参考实现
- **位置**: `qiqi/seqHMM-main/`
- **功能**: 一个功能完整的 R 包，专门为社交序列数据设计
- **主要特点**:
  - 支持多种 HMM 变体（基础 HMM、混合 HMM、非齐次 HMM 等）
  - 支持多通道（multichannel）序列数据
  - 支持协变量（covariates）建模
  - 提供完整的可视化功能
  - 使用 C++ 实现核心算法，性能较好

#### hmmlearn (Python库) - 可用基础库
- **位置**: `qiqi/hmmlearn-main/`
- **功能**: Python 生态系统中成熟的 HMM 实现库
- **主要特点**:
  - 提供基础的 HMM 实现（Categorical, Gaussian, Multinomial, Poisson, GMM）
  - 支持 EM 算法进行参数估计
  - 提供 Viterbi 解码和 forward-backward 算法
  - **局限性**: 不支持混合 HMM、非齐次 HMM 等高级功能

#### Sequenzo (当前项目)
- **位置**: `sequenzo/`
- **当前状态**: 
  - ✅ 已有完整的序列分析功能（聚类、可视化、距离度量等）
  - ✅ 已有事件历史分析（SAMM）功能
  - ❌ **缺少 HMM 相关功能**

---

## 🎯 seqHMM 提供的核心功能

### 基础模型类型

1. **基础 HMM (Hidden Markov Model)**
   - 标准的隐马尔可夫模型
   - 单通道或多通道序列数据
   - 功能: `build_hmm()`, `fit_model()`

2. **混合 HMM (Mixture HMM, MHMM)**
   - 多个 HMM 子模型的混合
   - 每个子模型代表一个聚类/类型
   - 支持协变量解释聚类成员关系
   - 功能: `build_mhmm()`, `fit_model()`

3. **非齐次 HMM (Non-homogeneous HMM, NHMM)**
   - 转移概率和发射概率可以随时间或协变量变化
   - 支持协变量建模（初始概率、转移概率、发射概率）
   - 功能: `build_nhmm()`, `fit_nhmm()`

4. **混合非齐次 HMM (Mixture NHMM, MNHMM)**
   - 结合混合模型和非齐次模型的优势
   - 功能: `build_mnhmm()`, `fit_mnhmm()`

5. **反馈增强 HMM (Feedback-augmented NHMM, FAN-HMM)**
   - 支持反馈机制的非齐次 HMM
   - 用于因果推断
   - 功能: `build_fanhmm()`

6. **其他变体**
   - Markov Models (MM) - 无隐层的马尔可夫模型
   - Mixture Markov Models (MMM)
   - Latent Class Models (LCM)

### 核心功能模块

1. **模型构建** (`build_*.R`)
   - 创建模型对象
   - 设置初始参数
   - 支持自定义初始值或随机初始化

2. **参数估计** (`fit_model.R`, `fit_*.R`)
   - EM 算法（主要方法）
   - 全局优化（可选）
   - 局部优化（可选）
   - 支持多线程并行计算

3. **预测和推断**
   - `predict()`: 预测隐藏状态序列
   - `posterior_probs()`: 计算后验概率
   - `hidden_paths()`: 获取最可能的隐藏路径
   - `forward_backward()`: Forward-Backward 算法

4. **模型评估**
   - `logLik()`: 对数似然值
   - `AIC()`, `BIC()`: 信息准则
   - `summary()`: 模型摘要
   - `vcov()`: 方差-协方差矩阵

5. **可视化**
   - `plot.hmm()`: HMM 可视化
   - `plot.mhmm()`: 混合 HMM 可视化
   - `stacked_sequence_plot()`: 堆叠序列图
   - `gridplot()`: 网格图

6. **数据模拟**
   - `simulate_hmm()`: 模拟 HMM 数据
   - `simulate_mhmm()`: 模拟混合 HMM 数据

---

## 💡 实现方案建议

### 方案一：基于 hmmlearn 扩展（推荐）

**优势**:
- ✅ 可以复用 hmmlearn 的基础架构（BaseHMM 类、EM 算法等）
- ✅ 减少重复代码
- ✅ 与 Python 生态系统兼容性好
- ✅ 可以逐步实现，先实现基础功能

**实现策略**:
1. **第一阶段：基础 HMM**
   - 基于 hmmlearn 的 `CategoricalHMM` 封装
   - 适配 Sequenzo 的 `SequenceData` 数据结构
   - 实现与 seqHMM 类似的 API 接口

2. **第二阶段：混合 HMM (MHMM)**
   - 实现多个 HMM 子模型的混合
   - 实现聚类成员概率计算
   - 支持协变量建模（可选）

3. **第三阶段：非齐次 HMM (NHMM)**
   - 实现协变量对转移/发射概率的影响
   - 实现 Softmax 参数化
   - 实现梯度计算

4. **第四阶段：高级功能**
   - 混合非齐次 HMM
   - 反馈增强 HMM
   - 可视化功能

### 方案二：完全独立实现

**优势**:
- ✅ 完全控制实现细节
- ✅ 可以针对序列分析优化

**劣势**:
- ❌ 工作量大
- ❌ 需要重新实现基础算法
- ❌ 容易引入 bug

**不推荐此方案**，除非有特殊需求。

### 方案三：直接调用 R 代码（不推荐）

**劣势**:
- ❌ 需要 R 运行时环境
- ❌ 性能开销大（R-Python 接口）
- ❌ 用户体验差
- ❌ 不符合 Sequenzo 的 Python 原生设计理念

---

## 📊 工作量评估

### 基础 HMM 实现
- **工作量**: 中等（2-3 周）
- **任务**:
  - 封装 hmmlearn 的 CategoricalHMM
  - 实现 `build_hmm()` 类似接口
  - 适配 SequenceData 数据结构
  - 实现基础可视化
  - 编写测试和文档

### 混合 HMM (MHMM) 实现
- **工作量**: 较大（3-4 周）
- **任务**:
  - 实现多个 HMM 子模型管理
  - 实现混合概率计算
  - 实现 EM 算法的混合版本
  - 实现协变量支持（可选）
  - 实现模型比较和选择

### 非齐次 HMM (NHMM) 实现
- **工作量**: 大（4-6 周）
- **任务**:
  - 实现协变量到概率的映射（Softmax）
  - 实现梯度计算
  - 实现数值优化（可能需要 scipy.optimize）
  - 实现预测和推断功能
  - 处理数值稳定性问题

### 高级功能（MNHMM, FAN-HMM）
- **工作量**: 很大（6-8 周）
- **任务**:
  - 结合混合和非齐次模型
  - 实现反馈机制
  - 实现因果推断相关功能

### 可视化功能
- **工作量**: 中等（2-3 周）
- **任务**:
  - 实现 HMM 状态转移图
  - 实现混合 HMM 的可视化
  - 实现序列堆叠图
  - 集成到 Sequenzo 的可视化系统

### 测试和文档
- **工作量**: 持续（每个阶段都需要）
- **任务**:
  - 单元测试
  - 集成测试
  - 与 seqHMM 结果对比测试
  - API 文档
  - 教程和示例

---

## 🎯 推荐实施路径

### 阶段一：MVP（最小可行产品）- 4-6 周
**目标**: 实现基础 HMM 功能，满足基本需求

1. **基础 HMM** (2-3 周)
   - 基于 hmmlearn 封装
   - 实现 `build_hmm()` 和 `fit_model()`
   - 实现 `predict()` 和 `posterior_probs()`
   - 基础可视化

2. **测试和文档** (1-2 周)
   - 单元测试
   - 与 seqHMM 对比测试
   - 编写使用文档

3. **集成到 Sequenzo** (1 周)
   - 添加到 `sequenzo/__init__.py`
   - 创建示例 notebook

### 阶段二：混合 HMM - 3-4 周
**目标**: 实现混合 HMM，支持聚类分析

1. **混合 HMM 实现** (2-3 周)
2. **测试和优化** (1 周)

### 阶段三：非齐次 HMM - 4-6 周
**目标**: 实现协变量建模能力

1. **NHMM 实现** (3-4 周)
2. **测试和优化** (1-2 周)

### 阶段四：完善和优化 - 持续
**目标**: 添加高级功能，优化性能

---

## 🔧 技术考虑

### 数据结构
- **输入**: Sequenzo 的 `SequenceData` 对象
- **输出**: 自定义的 HMM 模型对象（类似 seqHMM 的 `hmm`, `mhmm` 类）

### 依赖库
- **必需**:
  - `hmmlearn`: 基础 HMM 实现
  - `numpy`: 数值计算
  - `pandas`: 数据处理
  - `scipy`: 优化算法（用于 NHMM）

- **可选**:
  - `matplotlib`: 可视化（Sequenzo 已有）
  - `networkx`: 状态转移图可视化
  - `scikit-learn`: 一些工具函数

### 性能优化
- 使用 NumPy 向量化操作
- 对于大规模数据，考虑使用 Cython 或 C++ 扩展（类似 seqHMM）
- 支持并行计算（多线程/多进程）

### 数值稳定性
- 使用对数空间计算（log-space）避免下溢
- 实现 log-sum-exp 技巧
- 注意 Softmax 的数值稳定性

---

## 📝 关键决策点

1. **是否完全实现所有 seqHMM 功能？**
   - **建议**: 先实现核心功能（HMM, MHMM, NHMM），其他功能根据需求决定

2. **API 设计：完全模仿 seqHMM 还是 Python 风格？**
   - **建议**: Python 风格，但保持概念一致性，方便 R 用户迁移

3. **性能要求？**
   - **建议**: 至少与 seqHMM 相当，理想情况下更快（利用 Python 生态优势）

4. **是否需要与 seqHMM 完全兼容？**
   - **建议**: 概念兼容即可，不需要完全一致（因为语言差异）

---

## 🚀 快速开始建议

如果想快速验证可行性，建议：

1. **先做一个简单的原型**（1-2 天）
   - 使用 hmmlearn 的 CategoricalHMM
   - 封装一个简单的接口
   - 在 Sequenzo 的示例数据上测试
   - 验证与 seqHMM 的结果是否接近

2. **评估结果**
   - 如果结果合理，继续完整实现
   - 如果差异较大，需要深入分析原因

3. **逐步扩展**
   - 从简单到复杂
   - 每个阶段都进行充分测试

---

## 📚 参考资料

- **seqHMM 文档**: `qiqi/seqHMM-main/README.md`
- **seqHMM 论文**: Helske & Helske (2019), Journal of Statistical Software
- **hmmlearn 文档**: https://hmmlearn.readthedocs.io/
- **Sequenzo 架构**: `developer/ARCHITECTURE_GUIDE.md`

---

## ⚠️ 注意事项

1. **数值精度**: Python 和 R 的数值计算可能有细微差异，需要充分测试
2. **算法实现**: seqHMM 使用 C++ 实现核心算法，我们需要确保 Python 实现足够高效
3. **测试数据**: 建议使用 seqHMM 的示例数据进行对比测试
4. **向后兼容**: 确保新功能不影响 Sequenzo 的现有功能

---

**文档创建时间**: 2025年1月
**最后更新**: 2025年1月
