# Sequenzo Tests

本目录包含 Sequenzo 包的测试套件。

## 测试文件说明

### 基础测试
- **test_basic.py** - 基本的包导入和版本测试
- **test_pam_and_kmedoids.py** - PAM 和 K-medoids 聚类算法测试

### 集成测试
- **test_quickstart_integration.py** - 基于 quickstart 教程的完整工作流程测试
  - 测试数据加载
  - 测试 SequenceData 对象创建
  - 测试各种可视化功能
  - 测试距离矩阵计算
  - 测试聚类分析
  - 测试聚类质量评估
  - 测试完整的用户工作流程

### OpenMP 测试
- **openmp/** - OpenMP 相关的测试和解决方案

## 运行测试

### 运行所有测试
```bash
pytest tests/ -v
```

### 运行特定测试文件
```bash
# 运行基础测试
pytest tests/test_basic.py -v

# 运行集成测试（推荐）
pytest tests/test_quickstart_integration.py -v -s

# 运行 PAM/K-medoids 测试
pytest tests/test_pam_and_kmedoids.py -v
```

### 运行单个测试函数
```bash
# 运行完整工作流程测试
pytest tests/test_quickstart_integration.py::test_complete_workflow -v -s

# 运行可视化测试
pytest tests/test_quickstart_integration.py::test_visualizations_no_save -v -s
```

## 测试环境要求

测试需要安装以下依赖：
```bash
pip install pytest
pip install sequenzo  # 或者使用 pip install -e . 进行开发模式安装
```

## CI/CD 中的测试

在 GitHub Actions 中，我们在构建完 wheel 后会自动运行：
1. 基本的导入测试
2. C++ 扩展加载验证
3. 完整的集成测试（test_quickstart_integration.py）

这确保构建的 wheel 包在实际使用场景中能够正常工作。

## 集成测试的意义

`test_quickstart_integration.py` 模拟了真实用户的使用场景：
- 加载数据集
- 创建序列对象
- 进行可视化
- 计算距离矩阵
- 执行聚类分析
- 评估聚类质量

这些测试帮助我们在发布前发现用户可能遇到的问题，而不是等到用户反馈才知道。

## 开发建议

当你修改代码后，建议运行：
```bash
# 快速测试
pytest tests/test_basic.py

# 完整测试
pytest tests/test_quickstart_integration.py -v -s
```

如果所有测试都通过，说明你的修改没有破坏现有功能。

