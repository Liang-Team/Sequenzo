# Sequenzo 项目结构说明

## 📁 整理后的目录结构

```
Sequenzo-main/
├── __init__.py                 # ✅ 重要！包的顶级初始化文件
├── setup.py                    # 包构建配置
├── pyproject.toml              # 现代包配置
├── README.md                   # 项目主要说明
├── LICENSE                     # 许可证
│
├── docs/                       # 📚 技术文档
│   ├── README.md              # 文档说明
│   ├── OPENMP_FIX_SUMMARY.md  # OpenMP修复报告
│   ├── OPENMP_ENHANCEMENT.md  # CI/CD增强指南
│   ├── WINDOWS_OPENMP_GUIDE.md # Windows用户指南
│   └── ARCHITECTURE_GUIDE.md  # 架构编译指南
│
├── tools/                      # 🔧 开发工具
│   ├── README.md              # 工具说明
│   ├── test_openmp.py         # 通用OpenMP检测
│   └── check_windows_openmp.py # Windows专用检测
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
├── Tutorials/                  # 📖 教程和示例
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

## 🎯 主要改进

### ✅ 已完成的整理
1. **创建了 `docs/` 目录** - 所有技术文档集中管理
2. **创建了 `tools/` 目录** - 开发和测试工具集中管理
3. **移动了 OpenMP 相关文件** - 不再散落在根目录
4. **添加了说明文档** - 每个目录都有 README.md

### 📋 文件分类说明

#### 🔧 tools/ - 开发工具
- `test_openmp.py` - **通用版本** (macOS/Linux/Windows)
- `check_windows_openmp.py` - **Windows专用** (详细检测)

#### 📚 docs/ - 技术文档
- 面向开发者的技术文档
- OpenMP支持的完整实现记录
- 平台特定的使用指南

#### 💡 使用建议

**对于Windows学生**:
```bash
python tools/check_windows_openmp.py
```

**对于macOS/Linux开发者**:
```bash
python tools/test_openmp.py
```

**查看文档**:
- 技术细节: `docs/OPENMP_FIX_SUMMARY.md`
- Windows指南: `docs/WINDOWS_OPENMP_GUIDE.md`
- 工具说明: `tools/README.md`

---

⚠️ **重要**: `__init__.py` 是包的核心文件，绝对不能删除！
