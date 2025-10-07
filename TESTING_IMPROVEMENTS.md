# 测试改进说明

## 问题背景

之前的 GitHub Actions 只测试了基本的包导入和 C++ 扩展加载，但没有测试用户的实际使用场景。这导致：
- CI 通过了，但用户实际使用时可能遇到各种问题
- 无法及时发现集成问题
- 缺乏对完整工作流程的验证

## 解决方案

### 1. 修复 macOS wheel 构建问题

**问题**: macOS Python 3.9 构建失败，因为缺少 `wheel` 包。

**修复**:
- 在 `.github/workflows/python-app.yml` 中为所有 macOS 版本安装 `wheel`（之前只为 3.12 安装）
- 在通用依赖安装步骤中也添加 `setuptools wheel`

**修改文件**: `.github/workflows/python-app.yml` (第 45-47 行, 第 88 行)

### 2. 新增完整的集成测试

**创建文件**: `tests/test_quickstart_integration.py`

这个测试文件基于 `Tutorials/01_quickstart.ipynb`，包含：

#### 测试内容
1. **数据加载测试** (`test_dataset_loading`)
   - 验证数据集列表
   - 验证数据加载功能

2. **SequenceData 创建测试** (`test_sequence_data_creation`)
   - 验证序列对象能正确创建
   - 验证状态和时间点数量

3. **可视化测试** (`test_visualizations_no_save`)
   - Index plot
   - Legend plot
   - Most frequent sequences
   - Mean time plot
   - State distribution
   - Modal state
   - Transition matrix

4. **距离矩阵计算测试** (`test_distance_matrix_computation`)
   - OM 方法
   - TRATE 替代矩阵
   - 自动 indel 设置

5. **聚类工作流程测试** (`test_clustering_workflow`)
   - 聚类对象创建
   - 树状图绘制

6. **聚类质量评估测试** (`test_cluster_quality_evaluation`)
   - 计算聚类质量指标
   - CQI 表格生成
   - 质量分数绘图

7. **聚类结果提取测试** (`test_cluster_results_and_membership`)
   - 成员关系提取
   - 聚类分布统计
   - 分布可视化

8. **分组可视化测试** (`test_grouped_visualizations`)
   - 按聚类分组的 index plot
   - 按聚类分组的 state distribution

9. **完整工作流程测试** (`test_complete_workflow`)
   - 模拟用户从头到尾的完整使用流程
   - 验证所有步骤能够串联起来

#### 设计特点
- 使用 `matplotlib.use('Agg')` 避免在 CI 中显示图形
- 不保存任何文件，避免污染仓库
- 测试真实的用户使用场景
- 覆盖 quickstart 教程中的所有关键功能

### 3. GitHub Actions 集成测试步骤

**修改文件**: `.github/workflows/python-app.yml` (第 310-332 行)

在构建 wheel 后添加新的测试步骤：
```yaml
- name: Run Integration Tests (Quickstart Workflow)
  run: |
    # 安装构建好的 wheel
    # 安装 pytest
    # 运行集成测试
    pytest tests/test_quickstart_integration.py -v -s
```

这确保：
- 构建的 wheel 能够正常安装
- 所有核心功能在真实环境中工作正常
- 在发布前发现潜在问题

### 4. 本地验证脚本

**创建文件**: `verify_installation.py`

一个独立的验证脚本，用户可以在本地快速检查安装是否正常：

```bash
python verify_installation.py
```

验证内容：
1. 基本导入
2. C++ 扩展加载
3. 数据加载
4. SequenceData 创建
5. 可视化功能
6. 距离矩阵计算
7. 聚类分析

### 5. 测试文档

**创建文件**: `tests/README.md`

详细说明：
- 各个测试文件的用途
- 如何运行测试
- 测试环境要求
- CI/CD 中的测试流程
- 开发建议

## 使用方法

### 本地开发测试

```bash
# 快速验证安装
python verify_installation.py

# 运行所有测试
pytest tests/ -v

# 只运行集成测试
pytest tests/test_quickstart_integration.py -v -s

# 运行特定测试
pytest tests/test_quickstart_integration.py::test_complete_workflow -v -s
```

### CI/CD 自动测试

每次推送代码或创建 PR 时，GitHub Actions 会自动：
1. 构建所有平台的 wheel
2. 运行基本导入测试
3. 运行完整的集成测试
4. 验证 OpenMP 支持

## 好处

### 对用户
- 更高的质量保证
- 更少的安装问题
- 更可靠的使用体验

### 对开发者
- 及早发现问题
- 减少用户反馈的 bug
- 更有信心发布新版本

### 对项目
- 更好的 CI/CD 流程
- 更完善的测试覆盖
- 更专业的软件工程实践

## 文件清单

### 新增文件
- `tests/test_quickstart_integration.py` - 完整集成测试
- `tests/README.md` - 测试说明文档
- `verify_installation.py` - 本地验证脚本
- `TESTING_IMPROVEMENTS.md` - 本文档

### 修改文件
- `.github/workflows/python-app.yml` - 修复 wheel 安装问题，添加集成测试

## 下一步建议

1. **增加更多测试场景**
   - 测试其他数据集
   - 测试其他距离度量方法（DHD, HAM, LCP 等）
   - 测试加权数据
   - 测试多域序列分析

2. **性能测试**
   - 添加性能基准测试
   - 确保新版本不会降低性能

3. **边界情况测试**
   - 空数据
   - 缺失值处理
   - 极端参数值

4. **文档测试**
   - 验证文档中的代码示例能运行
   - 使用 doctest 测试 API 文档

5. **覆盖率报告**
   - 添加代码覆盖率工具（如 `pytest-cov`）
   - 在 CI 中生成覆盖率报告

## 总结

通过这些改进，Sequenzo 现在拥有：
- ✅ 更可靠的构建流程
- ✅ 更全面的测试覆盖
- ✅ 更真实的用户场景测试
- ✅ 更容易的本地验证
- ✅ 更好的开发者体验

这将大大减少"CI 通过但用户遇到问题"的情况！

