#!/bin/bash
# Sequenzo 用户诊断脚本
# 让用户运行这个脚本并把结果发给你

echo "======================================================================"
echo "Sequenzo 诊断脚本"
echo "======================================================================"
echo ""

# 1. 系统信息
echo "【1】系统信息："
python --version
echo "Python 路径: $(which python)"
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda 环境: $CONDA_DEFAULT_ENV"
fi
echo ""

# 2. 测试 NumPy
echo "【2】测试 NumPy："
python << 'NUMPY_TEST'
try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
    print(f"  路径: {numpy.__file__}")
except Exception as e:
    print(f"✗ NumPy 导入失败: {e}")
    print("\n>>> 问题: NumPy 安装损坏！")
    print(">>> 解决: pip uninstall numpy -y && pip install 'numpy>=2.0.0'")
    import sys
    sys.exit(1)
NUMPY_TEST

if [ $? -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "诊断结果: NumPy 安装有问题"
    echo "======================================================================"
    exit 1
fi

echo ""

# 3. 检查多个 NumPy
echo "【3】检查是否有多个 NumPy："
echo "Pip list:"
pip list | grep -i numpy
if command -v conda &> /dev/null; then
    echo "Conda list:"
    conda list numpy 2>/dev/null || echo "  (不在 conda 环境中)"
fi
echo ""

# 4. 测试 Sequenzo
echo "【4】测试 Sequenzo："
python << 'SEQUENZO_TEST'
try:
    import sequenzo
    print("✓ Sequenzo 导入成功")
    try:
        print(f"  版本: {sequenzo.__version__}")
    except:
        print("  版本: (开发版)")
    print(f"  路径: {sequenzo.__file__}")
except Exception as e:
    print(f"✗ Sequenzo 导入失败: {e}")
    print("\n>>> 问题: Sequenzo 安装或兼容性问题")
    print("\n完整错误信息：")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)
SEQUENZO_TEST

if [ $? -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "诊断结果: Sequenzo 有问题（但 NumPy 正常）"
    echo "======================================================================"
    echo ""
    echo "建议的解决方案："
    echo "1. 重装 sequenzo:"
    echo "   pip uninstall sequenzo -y"
    echo "   pip install --no-cache-dir sequenzo"
    echo ""
    echo "2. 如果还不行，从源码安装："
    echo "   pip install --no-binary sequenzo sequenzo"
    echo ""
    exit 1
fi

echo ""
echo "======================================================================"
echo "诊断结果: 一切正常！✓"
echo "======================================================================"

