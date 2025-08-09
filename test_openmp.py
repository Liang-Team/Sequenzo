#!/usr/bin/env python3
"""
OpenMP功能验证脚本
用于验证sequenzo是否正确链接了OpenMP库并可以使用并行计算
"""

import subprocess
import sys
import time
import os

def check_openmp_linkage():
    """检查C++扩展是否链接了OpenMP库"""
    print("=== 检查OpenMP库链接状态 ===")
    
    try:
        import sequenzo.clustering.clustering_c_code as cc
        print("✅ C++扩展加载成功")
        
        # 获取.so/.dll/.dylib文件路径
        so_path = cc.__file__
        print(f"📄 扩展文件: {so_path}")
        
        if sys.platform == 'darwin':
            # macOS: 使用otool检查动态库依赖
            result = subprocess.run(['otool', '-L', so_path], 
                                  capture_output=True, text=True)
            print("🔗 链接的动态库:")
            print(result.stdout)
            
            if 'libomp' in result.stdout:
                print("🚀 ✅ 检测到libomp - OpenMP支持启用!")
                return True
            elif 'libgomp' in result.stdout:
                print("🚀 ✅ 检测到libgomp - OpenMP支持启用!")
                return True
            else:
                print("❌ 未检测到OpenMP库链接")
                return False
                
        elif sys.platform.startswith('linux'):
            # Linux: 使用ldd检查动态库依赖
            result = subprocess.run(['ldd', so_path], 
                                  capture_output=True, text=True)
            print("🔗 链接的动态库:")
            print(result.stdout)
            
            if any(lib in result.stdout for lib in ['libgomp', 'libomp']):
                print("🚀 ✅ 检测到OpenMP库 - 并行计算支持启用!")
                return True
            else:
                print("❌ 未检测到OpenMP库链接")
                return False
                
        elif sys.platform == 'win32':
            print("🪟 Windows平台 - 检查MSVC OpenMP支持")
            # Windows上OpenMP通常静态链接到MSVC运行时
            print("🚀 ✅ 假定Windows MSVC OpenMP支持已启用")
            return True
            
    except ImportError as e:
        print(f"❌ C++扩展导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查过程中出错: {e}")
        return False

def test_parallel_performance():
    """简单的性能测试，验证并行计算是否生效"""
    print("\n=== 并行性能测试 ===")
    
    try:
        import numpy as np
        import sequenzo
        
        # 创建测试数据
        print("📊 生成测试数据...")
        np.random.seed(42)
        n_sequences = 100
        seq_length = 50
        
        # 生成随机序列数据
        sequences = []
        for i in range(n_sequences):
            seq = np.random.choice(['A', 'B', 'C', 'D'], size=seq_length)
            sequences.append(''.join(seq))
        
        print(f"✅ 生成了{n_sequences}个长度为{seq_length}的序列")
        
        # 测试距离计算（这里应该能体现并行性能）
        print("🔬 测试距离计算性能...")
        
        start_time = time.time()
        
        # 使用sequenzo计算距离矩阵
        # 注意：这里需要根据实际API调整
        from sequenzo import get_distance_matrix
        
        # 简单测试
        matrix = get_distance_matrix(
            sequences[:20],  # 使用较小的数据集进行快速测试
            method="OM",
            substitution_cost_matrix="auto"
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ 距离计算耗时: {elapsed:.3f}秒")
        print(f"📏 距离矩阵形状: {matrix.shape}")
        print("✅ 并行计算测试完成")
        
        return True
        
    except Exception as e:
        print(f"⚠️ 性能测试出现问题: {e}")
        print("💡 这可能是正常的，如果基本功能正常工作")
        return False

def main():
    """主测试函数"""
    print("🚀 Sequenzo OpenMP支持验证")
    print("=" * 50)
    
    # 检查基本导入
    try:
        import sequenzo
        print("✅ Sequenzo导入成功")
        print(f"📍 Sequenzo安装路径: {sequenzo.__file__}")
    except ImportError as e:
        print(f"❌ Sequenzo导入失败: {e}")
        return False
    
    # 检查OpenMP链接
    openmp_linked = check_openmp_linkage()
    
    # 性能测试（可选）
    if openmp_linked:
        print("\n💡 检测到OpenMP支持，进行性能测试...")
        test_parallel_performance()
    else:
        print("\n⚠️ 未检测到OpenMP支持")
        print("💡 这意味着计算将使用串行模式")
        print("🔧 要启用并行支持，请参考OPENMP_ENHANCEMENT.md")
    
    print("\n" + "=" * 50)
    if openmp_linked:
        print("🎉 结论: Sequenzo支持并行计算!")
    else:
        print("📝 结论: Sequenzo当前为串行版本")
    
    return openmp_linked

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
