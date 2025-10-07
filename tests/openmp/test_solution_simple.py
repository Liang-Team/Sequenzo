#!/usr/bin/env python3
"""
简化的 Apple Silicon OpenMP 解决方案测试

这个脚本测试新实现的核心功能。
"""

import sys
import os
import platform
from pathlib import Path


def test_files_exist():
    """测试必要文件是否存在"""
    print("🧪 测试文件存在性...")
    
    files_to_check = [
        "sequenzo/openmp_setup.py",
        "scripts/post_install.py", 
        "APPLE_SILICON_GUIDE.md",
        "setup.py"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist


def test_openmp_setup_content():
    """测试 OpenMP 设置模块内容"""
    print("\n🧪 测试 OpenMP 设置模块内容...")
    
    setup_file = Path(__file__).parent / "sequenzo" / "openmp_setup.py"
    
    if not setup_file.exists():
        print("❌ OpenMP 设置文件不存在")
        return False
    
    try:
        with open(setup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键函数是否存在
        required_functions = [
            'ensure_openmp_support',
            'get_openmp_status', 
            'check_libomp_availability',
            'check_homebrew_available'
        ]
        
        all_functions_exist = True
        for func in required_functions:
            if f'def {func}(' in content:
                print(f"✅ 函数 {func} 存在")
            else:
                print(f"❌ 函数 {func} 不存在")
                all_functions_exist = False
        
        return all_functions_exist
        
    except Exception as e:
        print(f"❌ 读取文件时出现错误: {e}")
        return False


def test_setup_integration():
    """测试 setup.py 集成"""
    print("\n🧪 测试 setup.py 集成...")
    
    setup_file = Path(__file__).parent / "setup.py"
    
    if not setup_file.exists():
        print("❌ setup.py 不存在")
        return False
    
    try:
        with open(setup_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键内容是否存在
        required_content = [
            'install_libomp_on_apple_silicon',
            'from sequenzo.openmp_setup import',
            'InstallCommand',
            'post_install.py'
        ]
        
        all_content_exists = True
        for item in required_content:
            if item in content:
                print(f"✅ {item} 存在")
            else:
                print(f"❌ {item} 不存在")
                all_content_exists = False
        
        return all_content_exists
        
    except Exception as e:
        print(f"❌ 读取文件时出现错误: {e}")
        return False


def test_platform_detection():
    """测试平台检测"""
    print("\n🧪 测试平台检测...")
    
    print(f"   - 平台: {sys.platform}")
    print(f"   - 架构: {platform.machine()}")
    print(f"   - 系统: {platform.system()}")
    
    is_darwin = sys.platform == 'darwin'
    is_arm64 = platform.machine() == 'arm64'
    
    print(f"   - macOS: {is_darwin}")
    print(f"   - ARM64: {is_arm64}")
    
    if is_darwin and is_arm64:
        print("✅ 检测到 Apple Silicon Mac")
    else:
        print("ℹ️  非 Apple Silicon Mac")
    
    return True


def main():
    """主测试函数"""
    print("🚀 Apple Silicon OpenMP 解决方案测试（简化版）")
    print("=" * 60)
    
    # 测试结果
    results = []
    
    # 测试文件存在性
    results.append(test_files_exist())
    
    # 测试 OpenMP 设置模块内容
    results.append(test_openmp_setup_content())
    
    # 测试 setup.py 集成
    results.append(test_setup_integration())
    
    # 测试平台检测
    results.append(test_platform_detection())
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   - 通过: {passed}/{total}")
    print(f"   - 失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！解决方案已就绪。")
        print("\n💡 下一步:")
        print("   1. 在 Apple Silicon Mac 上测试安装")
        print("   2. 验证自动 OpenMP 安装功能")
        print("   3. 测试并行计算性能")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
