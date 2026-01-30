# 快速回答：为什么 .c 文件不能 commit？

## 简短回答

**`.pyx` 文件（源代码）：** 
- ✅ 应该 commit  
- 你写的代码，人类可读，平台无关

**`.c` 文件（Cython 生成）：**
- ❌ 不应该 commit  
- 机器自动生成，包含**你电脑的硬编码路径**
- 用户电脑上路径不一样 → 编译失败

## 可视化说明

```
你的开发流程：
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  seqconc.pyx│ Cython  │ seqconc.c   │ Compiler│ seqconc.so  │
│  (源代码)    │ ──────> │ (C 代码)     │ ──────> │ (二进制)     │
│  ✅ commit   │         │ ❌ 不commit  │         │ ❌ 不commit  │
└─────────────┘         └─────────────┘         └─────────────┘
                              ↓
                        包含你的路径：
                        /Users/lei/Documents/.../
```

## 为什么以前没问题？

### 1. 可能一直有问题，但被忽略了
- 少数用户遇到 → 自己解决或放弃
- 没有反馈

### 2. 最近环境变化更大
- **NumPy 2.0**（2024）→ 更严格的版本检查
- **Apple Silicon (M1/M2/M3)** → 架构不同（arm64 vs x86_64）  
- **Python 3.12** → 更多变化
- 用户环境更多样化

### 3. 这次的 .c 文件特别"糟糕"

看你这次的 `.c` 文件：

```c
"include_dirs": [
    "/Users/lei/Documents/Sequenzo_all_folders/sequence_data_sources/.conda/lib/python3.12/..."
]
"extra_compile_args": [
    "-arch", "x86_64",  // 只支持 Intel Mac！
]
```

**问题：**
- 包含**完整的绝对路径**（之前可能是相对路径）
- 指定**特定架构** x86_64（不支持 Apple Silicon）
- 指向**特定 conda 环境**（用户可能没有 conda）

用户的电脑：
- Windows → 路径格式完全不同（`C:\Users\...`）
- ARM Mac → 需要 `arm64`，不是 `x86_64`
- 没有 conda → 路径根本不存在

## 应该 commit 什么？

| 类型 | 文件 | Commit? | 为什么？ |
|------|------|---------|---------|
| 源代码 | `seqconc.pyx` | ✅ | 你写的，平台无关 |
| 源代码 | `OMdistance.cpp` | ✅ | 你写的，平台无关 |
| 源代码 | `utils.h` | ✅ | 你写的，平台无关 |
| 生成代码 | `seqconc.c` | ❌ | Cython 生成，含硬编码路径 |
| 二进制 | `seqconc.so` | ❌ | 编译产物，平台特定 |
| 二进制 | `c_code.so` | ❌ | 编译产物，平台特定 |

## 正确的流程

### 开发时（你）
```bash
vim seqconc.pyx              # 修改源代码
python setup.py build_ext    # 本地测试（生成 .c 和 .so）
git add seqconc.pyx          # 只 commit .pyx
git commit -m "优化性能"
```

### 发布时（CI）
```bash
# GitHub Actions 自动执行
cython seqconc.pyx → seqconc.c    # 在 CI 环境重新生成 .c
gcc seqconc.c → seqconc.so        # 为每个平台编译
# 打包到 wheel 分发
```

### 用户安装（用户）
```bash
pip install sequenzo
# 下载预编译的 wheel（包含编译好的 .so）
# 或者从源码安装（自动运行 Cython + 编译）
```

## 现在的修复

```bash
# 1. 从 git 删除这些文件
git rm --cached sequenzo/dissimilarity_measures/utils/*.c  ✅ 已完成
git rm --cached sequenzo/big_data/clara/utils/*.c          # 也需要删除

# 2. 更新 .gitignore
# 确保 *.c 被忽略（Cython 生成的）
# 但保留手写的 .cpp（Pybind11 源码）  ✅ 已完成

# 3. Commit 并发布新版本
git commit -m "Fix: Remove Cython-generated .c files"
# 推送后，CI 会重新生成干净的 wheel
```

## 用户遇到问题怎么办？

### 当前版本（v0.1.19）- 有问题
用户应该：
```bash
pip install --upgrade "numpy>=2.0.0"  # 升级 numpy
pip install --no-cache-dir sequenzo   # 重装
```

### 下一版本（v0.1.20+）- 修复后
- `.c` 文件不再在 git 里
- CI 会为每个平台重新生成
- 用户不会遇到这个问题

---

## 总结一句话

**`.c` 文件是机器生成的，包含你电脑的特定信息（路径、架构），其他人的电脑用不了。只应该 commit 人类写的源代码（`.pyx`, `.cpp`），让机器在需要时自动生成 `.c` 文件。**

