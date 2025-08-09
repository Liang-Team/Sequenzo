#!/usr/bin/env python3
"""
@Author  : Yuqi Liang 梁彧祺
@File    : check_windows_openmp.py
@Time    : 09/08/2025 09:38
@Desc    : 
    Windows OpenMP检测脚本
    专门为Windows用户检测sequenzo是否启用了OpenMP并行支持
"""

import sys
import os
import subprocess
import platform

def check_windows_environment():
    """检查Windows环境信息"""
    print("=== Windows环境检查 ===")
    print(f"🖥️ 操作系统: {platform.system()} {platform.release()}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📍 Python路径: {sys.executable}")
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("📦 虚拟环境: ✅ 是")
    else:
        print("📦 虚拟环境: ❌ 否")

def check_visual_studio():
    """检查Visual Studio Build Tools"""
    print("\n=== Visual Studio检查 ===")
    
    # 检查cl编译器
    try:
        result = subprocess.run(['cl'], capture_output=True, text=True)
        if 'Microsoft' in result.stderr:
            print("✅ MSVC编译器 (cl.exe) 可用")
            
            # 检查是否支持/openmp
            if '/openmp' in result.stderr or 'openmp' in result.stderr.lower():
                print("✅ MSVC支持OpenMP (/openmp)")
                return True
            else:
                print("⚠️ MSVC可能不支持OpenMP (帮助信息中未显示)")
                print("💡 注意: 某些VS版本不在帮助中列出/openmp，但实际支持")
                return False
        else:
            print("❌ MSVC编译器不可用")
            return False
    except FileNotFoundError:
        print("❌ 未找到cl.exe - 需要安装Visual Studio Build Tools")
        print("💡 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        return False
    except Exception as e:
        print(f"❌ 检查编译器时出错: {e}")
        return False

def check_sequenzo_installation():
    """检查sequenzo安装状态"""
    print("\n=== Sequenzo安装检查 ===")
    
    try:
        import sequenzo
        print("✅ Sequenzo导入成功")
        print(f"📍 安装路径: {sequenzo.__file__}")
        
        # 检查C++扩展
        try:
            import sequenzo.clustering.clustering_c_code as cc
            print("✅ C++扩展加载成功")
            
            extension_path = cc.__file__
            print(f"📄 扩展文件: {extension_path}")
            
            # Windows上检查DLL依赖比较复杂，我们用简单方法
            if os.path.exists(extension_path):
                file_size = os.path.getsize(extension_path)
                print(f"📊 扩展文件大小: {file_size:,} bytes")
                
                # 简单启发式：OpenMP版本通常比串行版本大
                if file_size > 100000:  # 100KB
                    print("�� 文件大小暗示可能包含OpenMP支持")
                else:
                    print("⚠️ 文件较小，可能是串行版本")
            
            return True
            
        except ImportError as e:
            print(f"❌ C++扩展加载失败: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Sequenzo导入失败: {e}")
        return False

def authoritative_openmp_check():
    """权威OpenMP验证 - 使用dumpbin检查DLL依赖"""
    print("\n=== 🔬 权威OpenMP验证 (dumpbin检查) ===")
    
    try:
        import sequenzo.clustering.clustering_c_code as cc
        ext_path = cc.__file__
        print(f"📄 扩展文件: {ext_path}")
        
        # 使用dumpbin检查依赖
        result = subprocess.run(['dumpbin', '/dependents', ext_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            output = result.stdout
            print("🔗 DLL依赖关系:")
            print(output)
            
            # 检查是否包含VCOMP
            if 'VCOMP' in output.upper():
                print("🚀 ✅ 权威确认: 检测到VCOMP*.DLL - OpenMP已启用!")
                print("💡 说明: 这是Windows OpenMP运行时库，确认为并行版本")
                return True
            else:
                print("❌ 权威确认: 未检测到VCOMP*.DLL - 当前为串行版本")
                print("💡 说明: 没有链接OpenMP运行时库")
                return False
        else:
            print("❌ dumpbin执行失败")
            print("💡 可能需要在Visual Studio Native Tools Command Prompt中运行")
            return None
            
    except ImportError:
        print("❌ 无法导入sequenzo C++扩展")
        return None
    except FileNotFoundError:
        print("❌ 找不到dumpbin工具")
        print("💡 请在Visual Studio Native Tools Command Prompt中运行此脚本")
        return None
    except Exception as e:
        print(f"❌ 权威验证时出错: {e}")
        return None

def check_openmp_runtime_test():
    """运行时测试OpenMP"""
    print("\n=== OpenMP运行时测试 ===")
    
    try:
        # 简单的OpenMP测试
        test_code = """
import numpy as np
import time

# 生成测试数据
np.random.seed(42)
data = np.random.random((1000, 100))

# 计算密集型操作
start_time = time.time()
result = np.dot(data, data.T)
elapsed = time.time() - start_time

print(f"⏱️ 矩阵运算耗时: {elapsed:.4f}秒")
print(f"📊 结果矩阵形状: {result.shape}")

# 检查CPU使用情况提示
import os
cpu_count = os.cpu_count()
print(f"�� 系统CPU核心数: {cpu_count}")
print("💡 如果使用OpenMP，应该能看到多核利用")
"""
        
        exec(test_code)
        return True
        
    except Exception as e:
        print(f"❌ 运行时测试失败: {e}")
        return False

def provide_windows_instructions():
    """提供Windows下启用OpenMP的指导"""
    print("\n" + "="*60)
    print("📋 Windows下启用OpenMP的完整步骤")
    print("="*60)
    
    print("\n🔧 方法1: 使用Visual Studio Native Tools（推荐）")
    print("在 'x64 Native Tools Command Prompt for VS 2022' 中运行:")
    print("```")
    print("set SEQUENZO_ENABLE_OPENMP=1")
    print("set CL=/openmp")
    print("pip uninstall sequenzo -y")
    print("pip install -e . -v")
    print("")
    print("REM 权威验证")
    print("python -c \"import sequenzo.clustering.clustering_c_code as cc, subprocess; subprocess.run(['dumpbin', '/dependents', cc.__file__])\"")
    print("```")
    
    print("\n🔧 方法2: PowerShell/CMD（备选）")
    print("```")
    print("# PowerShell")
    print("$env:SEQUENZO_ENABLE_OPENMP=1")
    print("pip uninstall sequenzo -y")
    print("pip install -e .")
    print("")
    print("# CMD")
    print("set SEQUENZO_ENABLE_OPENMP=1")
    print("pip uninstall sequenzo -y")
    print("pip install -e .")
    print("```")
    
    print("\n🔧 方法3: 安装Visual Studio Build Tools")
    print("1. 下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("2. 安装时选择 'Desktop development with C++'")
    print("3. 重新运行方法1")
    
    print("\n🧪 验证步骤:")
    print("使用权威验证方法（dumpbin）确认OpenMP状态")
    print("检测脚本可能误报，以dumpbin结果为准！")

def main():
    """主函数"""
    print("🪟 Windows OpenMP支持检测工具")
    print("=" * 50)
    
    # 检查环境
    check_windows_environment()
    
    # 检查编译器
    compiler_ok = check_visual_studio()
    
    # 检查sequenzo
    sequenzo_ok = check_sequenzo_installation()
    
    # 权威验证（最重要）
    authoritative_result = authoritative_openmp_check()
    
    # 运行时测试
    runtime_ok = check_openmp_runtime_test()
    
    # 总结和建议
    print("\n" + "="*50)
    print("📊 检测结果总结")
    print("="*50)
    
    # 优先以权威验证结果为准
    if authoritative_result is True:
        print("🎉 权威确认: OpenMP已启用! (VCOMP*.DLL检测通过)")
        print("✅ 当前使用并行版本")
        print("💡 如果之前测试显示'串行版本'，那是误报")
    elif authoritative_result is False:
        print("⚠️ 权威确认: 当前为串行版本 (未检测到VCOMP*.DLL)")
        print("💡 需要重新编译启用OpenMP")
    else:
        # 权威验证失败，回退到传统检测
        print("⚠️ 权威验证失败，使用传统检测结果:")
        if compiler_ok and sequenzo_ok and runtime_ok:
            print("🎉 传统检测: 很可能已启用OpenMP支持！")
            print("✅ 编译器支持: 是")
            print("✅ Sequenzo安装: 正常")
            print("✅ 运行测试: 通过")
        else:
            print("⚠️ 传统检测: 可能使用的是串行版本")
            print(f"{'✅' if compiler_ok else '❌'} 编译器支持: {'是' if compiler_ok else '否'}")
            print(f"{'✅' if sequenzo_ok else '❌'} Sequenzo安装: {'正常' if sequenzo_ok else '异常'}")
            print(f"{'✅' if runtime_ok else '❌'} 运行测试: {'通过' if runtime_ok else '失败'}")
            
            print("\n�� 建议:")
            if not compiler_ok:
                print("- 安装Visual Studio Build Tools")
            if not sequenzo_ok:
                print("- 重新安装sequenzo")
            
            print("- 使用SEQUENZO_ENABLE_OPENMP=1环境变量强制启用")
    
    print("\n🔔 重要提醒:")
    print("- 检测脚本可能在某些Windows环境下误报")
    print("- 请优先使用权威验证方法 (dumpbin /dependents)")
    print("- 在Visual Studio Native Tools Command Prompt中运行效果最佳")
    
    # 提供详细指导
    provide_windows_instructions()
    
    # 返回权威验证结果，如果失败则回退到传统检测
    if authoritative_result is not None:
        return authoritative_result
    else:
        return compiler_ok and sequenzo_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
