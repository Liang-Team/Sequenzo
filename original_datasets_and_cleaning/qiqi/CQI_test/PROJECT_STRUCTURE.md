# Sequenzo 项目结构说明

## 📁 整理后的目录结构

```
Sequenzo-main/
├── __init__.py                 # ✅ 包的顶级初始化文件
├── setup.py                    # 包构建配置
├── pyproject.toml              # 现代包配置
├── README.md                   # 项目主要说明（用户文档）
├── LICENSE                     # 许可证
│
├── developer/                  # 🔧 开发者工具包（内部专用）
│   ├── README.md              # 开发者指南
│   ├── test_openmp.py         # 通用OpenMP检测工具
│   ├── check_windows_openmp.py # Windows专用检测工具
│   ├── OPENMP_FIX_SUMMARY.md  # OpenMP修复技术报告
│   ├── OPENMP_ENHANCEMENT.md  # CI/CD增强指南
│   ├── WINDOWS_OPENMP_GUIDE.md # Windows开发指南
│   └── ARCHITECTURE_GUIDE.md  # macOS架构编译指南
│
├── sequenzo/                   # 📦 主要包代码
│   ├── __init__.py            # 包初始化
│   ├── clustering/            # 聚类算法
│   ├── dissimilarity_measures/ # 距离计算
│   ├── data_preprocessing/    # 数据预处理
│   ├── visualization/         # 可视化
│   ├── datasets/             # 内置数据集
│   ├── big_data/             # 大数据处理
│   ├── multidomain/          # 多域分析
│   ├── prefix_tree/          # 前缀树
│   └── suffix_tree/          # 后缀树
│
├── tests/                      # 🧪 单元测试
│   ├── test_basic.py
│   └── test_pam_and_kmedoids.py
│
├── Tutorials/                  # 📖 用户教程和示例
│   ├── 01_quickstart.ipynb
│   └── ...
│
├── .github/                    # 🤖 CI/CD配置
│   └── workflows/
│       └── python-app.yml
│
├── assets/                     # 🎨 资源文件
│   └── logo/
│
├── requirements-*.txt          # 依赖文件
├── venv/                      # 虚拟环境
├── build/                     # 构建临时文件
├── dist/                      # 分发包
└── original_datasets_and_cleaning/ # 原始数据和清理脚本
```

## 🎯 重要改进: 统一开发者材料

### ✅ 解决的问题
1. **避免误解**: 之前的 `docs/` 容易被误认为用户文档
2. **逻辑统一**: 工具和文档本质上是"连体婴"，都是开发者专用
3. **使用便利**: 所有开发相关的材料现在都在一个地方

### 📂 目录定位说明

#### 🔧 `developer/` - 开发者专用工具包
**受众**: 开发者、维护者、贡献者  
**性质**: 内部开发材料，不面向最终用户  
**内容**:
- OpenMP检测和诊断工具
- 技术实施文档和故障排查指南
- 平台特定的开发配置指南
- CI/CD配置和维护文档

#### 📖 `Tutorials/` - 用户教程
**受众**: 最终用户、研究人员、学习者  
**性质**: 面向用户的学习材料  
**内容**:
- 快速入门教程
- 功能演示示例
- 使用场景指南

#### 🧪 `tests/` - 自动化测试
**受众**: 开发者、CI/CD系统  
**性质**: 质量保证和回归测试  

## 💡 使用指南

### 🧑‍💻 开发者工作流
```bash
# 1. 环境设置和验证
python developer/test_openmp.py          # 通用检测
python developer/check_windows_openmp.py # Windows专用

# 2. 了解技术架构
less developer/OPENMP_FIX_SUMMARY.md

# 3. 解决平台特定问题
less developer/WINDOWS_OPENMP_GUIDE.md   # Windows
less developer/ARCHITECTURE_GUIDE.md     # macOS
```

### 👥 用户学习路径
```bash
# 用户从这里开始学习
jupyter notebook Tutorials/01_quickstart.ipynb
```

### 🔧 维护者操作
```bash
# 查看CI/CD配置和维护指南
less developer/OPENMP_ENHANCEMENT.md
less .github/workflows/python-app.yml
```

## 🎉 结构优势

1. **📋 清晰分离**: 开发者材料 vs 用户材料
2. **🔍 易于发现**: 所有开发工具集中在一个位置
3. **🛠️ 便于维护**: 相关的工具和文档放在一起
4. **📚 逻辑统一**: 避免了人为的工具/文档分离

---

⚠️ **重要提醒**: 
- `developer/` 目录是**内部开发材料**，不是用户文档
- 用户文档主要在项目根目录的 `README.md` 和 `Tutorials/` 目录
- `__init__.py` 等核心文件请勿随意删除或移动
