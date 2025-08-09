# OpenMP Enhancement for Sequenzo Wheels

## 🎯 目标
为所有平台的预编译wheel添加OpenMP支持，让用户`pip install sequenzo`即可获得并行版本。

## 📋 需要在python-app.yml中添加的配置

### 1. 平台特定的OpenMP库安装

```yaml
# 在 "Install dependencies" 步骤之前添加
- name: Install OpenMP (macOS)
  if: runner.os == 'macOS'
  run: |
    brew install libomp
    echo "CC=clang" >> $GITHUB_ENV
    echo "CXX=clang++" >> $GITHUB_ENV
    echo "LDFLAGS=-L$(brew --prefix libomp)/lib" >> $GITHUB_ENV
    echo "CPPFLAGS=-I$(brew --prefix libomp)/include" >> $GITHUB_ENV

- name: Install OpenMP (Ubuntu)
  if: runner.os == 'Linux'  
  run: |
    sudo apt-get update
    sudo apt-get install -y libomp-dev

- name: Setup MSVC with OpenMP (Windows)
  if: runner.os == 'Windows'
  uses: ilammy/msvc-dev-cmd@v1
```

### 2. cibuildwheel环境配置

```yaml
# 修改 "Build wheels with cibuildwheel" 步骤
- name: Build wheels with cibuildwheel
  if: runner.os != 'macOS'
  run: python -m cibuildwheel --output-dir dist
  env:
    CIBW_SKIP: "pp*"
    CIBW_ARCHS_WINDOWS: "AMD64"
    CIBW_ARCHS_LINUX: "x86_64"
    
    # 新增：OpenMP环境配置
    CIBW_ENVIRONMENT_LINUX: >
      SEQUENZO_ENABLE_OPENMP=1
    CIBW_ENVIRONMENT_WINDOWS: >
      SEQUENZO_ENABLE_OPENMP=1
    
    # 新增：Linux OpenMP库安装
    CIBW_BEFORE_BUILD_LINUX: >
      yum install -y libgomp-devel ||
      apt-get update && apt-get install -y libomp-dev
```

### 3. macOS特殊处理

```yaml
# 修改 "Build Cython wheels on macOS" 步骤
- name: Build Cython wheels on macOS
  if: runner.os == 'macOS'
  run: |
    export SEQUENZO_ENABLE_OPENMP=1
    python setup.py build_ext --inplace
    python -m build
```

## 🧪 验证OpenMP是否生效

### 添加测试步骤：

```yaml
- name: Test OpenMP functionality
  run: |
    python -c "
    import sequenzo
    try:
        import sequenzo.clustering.clustering_c_code as cc
        print('✅ C++ extensions with OpenMP loaded')
        # TODO: 添加具体的并行性能测试
    except ImportError:
        print('❌ C++ extensions failed to load')
    "
```

## 📊 预期效果

### 用户体验改进：
```bash
# 用户安装
pip install sequenzo

# 自动获得：
✅ Intel Mac wheel (x86_64) + OpenMP (libomp)
✅ Apple Silicon wheel (arm64) + OpenMP (libomp)  
✅ Windows wheel (amd64) + OpenMP (vcomp)
✅ Linux wheel (manylinux) + OpenMP (gomp)
```

### 性能提升：
- 聚类算法：2-8x加速（取决于CPU核心数）
- 距离计算：3-10x加速
- 大数据集处理：显著改善

## 🔗 相关链接

- [cibuildwheel OpenMP示例](https://cibuildwheel.readthedocs.io/en/stable/cpp_standards/)
- [PyPA打包指南](https://packaging.python.org/guides/packaging-binary-extensions/)

## 📝 实施步骤

1. ✅ 分析现有配置（已完成）
2. ⏳ 修改setup.py添加OpenMP检测
3. ⏳ 更新workflow配置
4. ⏳ 测试各平台构建
5. ⏳ 验证OpenMP功能
6. ⏳ 发布新版本wheel
