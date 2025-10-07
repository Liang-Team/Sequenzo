#!/usr/bin/env python3
"""
测试 Apple Silicon OpenMP 自动安装解决方案

这个脚本用于测试新实现的自动 OpenMP 依赖管理功能。
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


def test_openmp_setup_module():
    """测试 OpenMP 设置模块"""
    print("🧪 测试 OpenMP 设置模块...")
    
    try:
        # 尝试导入 OpenMP 设置模块
        from sequenzo.openmp_setup import ensure_openmp_support, get_openmp_status
        
        print("✅ OpenMP 设置模块导入成功")
        
        # 获取状态信息
        status = get_openmp_status()
        print(f"📊 系统状态:")
        print(f"   - 平台: {status['platform']}")
        print(f"   - 架构: {status['architecture']}")
        print(f"   - Apple Silicon: {status['is_apple_silicon']}")
        print(f"   - libomp 可用: {status['libomp_available']}")
        print(f"   - Homebrew 可用: {status['homebrew_available']}")
        print(f"   - Conda 环境: {status['conda_environment']}")
        
        # 尝试确保 OpenMP 支持
        success = ensure_openmp_support()
        
        if success:
            print("✅ OpenMP 支持已就绪")
        else:
            print("⚠️  OpenMP 支持不可用")
        
        return success
        
    except ImportError as e:
        print(f"❌ 无法导入 OpenMP 设置模块: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False


def test_setup_integration():
    """测试 setup.py 集成"""
    print("\n🧪 测试 setup.py 集成...")
    
    try:
        # 检查 setup.py 是否可以正常导入
        import setup
        
        print("✅ setup.py 导入成功")
        
        # 检查关键函数是否存在
        if hasattr(setup, 'install_libomp_on_apple_silicon'):
            print("✅ install_libomp_on_apple_silicon 函数存在")
        else:
            print("❌ install_libomp_on_apple_silicon 函数不存在")
            return False
        
        if hasattr(setup, 'has_openmp_support'):
            print("✅ has_openmp_support 函数存在")
        else:
            print("❌ has_openmp_support 函数不存在")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ 无法导入 setup.py: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False


def test_post_install_script():
    """测试安装后脚本"""
    print("\n🧪 测试安装后脚本...")
    
    script_path = Path(__file__).parent / "scripts" / "post_install.py"
    
    if not script_path.exists():
        print("❌ 安装后脚本不存在")
        return False
    
    print("✅ 安装后脚本存在")
    
    try:
        # 尝试运行脚本
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 安装后脚本运行成功")
            print("📝 输出:")
            print(result.stdout)
            return True
        else:
            print(f"❌ 安装后脚本运行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 安装后脚本运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行安装后脚本时出现错误: {e}")
        return False


def test_apple_silicon_detection():
    """测试 Apple Silicon 检测"""
    print("\n🧪 测试 Apple Silicon 检测...")
    
    is_darwin = sys.platform == 'darwin'
    is_arm64 = platform.machine() == 'arm64'
    
    print(f"   - macOS: {is_darwin}")
    print(f"   - ARM64: {is_arm64}")
    print(f"   - 平台: {sys.platform}")
    print(f"   - 架构: {platform.machine()}")
    
    if is_darwin and is_arm64:
        print("✅ 检测到 Apple Silicon Mac")
        return True
    else:
        print("ℹ️  非 Apple Silicon Mac，跳过相关测试")
        return True


def main():
    """主测试函数"""
    print("🚀 Apple Silicon OpenMP 解决方案测试")
    print("=" * 50)
    
    # 测试结果
    results = []
    
    # 测试 Apple Silicon 检测
    results.append(test_apple_silicon_detection())
    
    # 测试 OpenMP 设置模块
    results.append(test_openmp_setup_module())
    
    # 测试 setup.py 集成
    results.append(test_setup_integration())
    
    # 测试安装后脚本
    results.append(test_post_install_script())
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   - 通过: {passed}/{total}")
    print(f"   - 失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！解决方案已就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
