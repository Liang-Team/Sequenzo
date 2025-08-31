#!/usr/bin/env python3
"""
环境测试脚本：验证基本功能是否正常
用于在运行大规模测试前检查环境
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入功能"""
    print("测试导入功能...")
    try:
        from sequenzo.define_sequence_data import SequenceData
        from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
        print("✓ 导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_small_dataset():
    """测试小数据集"""
    print("\n测试小数据集（100个序列，50个唯一）...")
    try:
        from sequenzo.define_sequence_data import SequenceData
        from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
        
        # 创建小数据集
        np.random.seed(42)
        states = [0, 1, 2, 3, 4]
        seq_length = 5
        
        # 生成唯一序列
        unique_seqs = []
        for _ in range(50):
            seq = np.random.choice(states, size=seq_length)
            unique_seqs.append(seq)
        
        # 通过重复生成总序列数
        all_seqs = []
        all_ids = []
        for i in range(100):
            seq_idx = i % 50
            all_seqs.append(unique_seqs[seq_idx])
            all_ids.append(f"seq_{i}")
        
        # 创建DataFrame
        df_data = {'worker_id': all_ids}
        for j in range(seq_length):
            df_data[f'C{j+1}'] = [seq[j] for seq in all_seqs]
        
        df = pd.DataFrame(df_data)
        
        # 创建SequenceData对象
        time_cols = [f'C{j+1}' for j in range(seq_length)]
        sequence_data = SequenceData(
            df, 
            time=time_cols, 
            time_type="age", 
            states=states, 
            id_col="worker_id"
        )
        
        print(f"✓ 创建数据集成功: {sequence_data.seqdata.shape}")
        
        # 测试距离矩阵计算
        start_time = time.time()
        result = get_distance_matrix(
            sequence_data, 
            method="OMspell", 
            sm="TRATE", 
            indel="auto",
            full_matrix=True
        )
        end_time = time.time()
        
        print(f"✓ 距离矩阵计算成功: {result.shape}")
        print(f"✓ 计算时间: {end_time - start_time:.2f} 秒")
        
        return True
        
    except Exception as e:
        print(f"✗ 小数据集测试失败: {e}")
        return False

def test_memory_usage():
    """测试内存使用情况"""
    print("\n测试内存使用情况...")
    try:
        # 尝试创建较大的数组来测试内存
        test_array = np.random.random((1000, 1000))
        memory_usage = test_array.nbytes / 1024 / 1024  # MB
        print(f"✓ 内存测试成功: 创建了 {memory_usage:.2f} MB 的数组")
        del test_array  # 释放内存
        return True
    except Exception as e:
        print(f"✗ 内存测试失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("get_distance_matrix 环境测试")
    print("="*60)
    
    tests = [
        ("导入功能", test_imports),
        ("小数据集", test_small_dataset),
        ("内存使用", test_memory_usage),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"⚠️  {test_name} 测试失败")
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境准备就绪，可以运行大规模测试。")
        print("\n建议运行顺序：")
        print("1. 先运行边界测试: python test_get_distance_matrix_simple.py")
        print("2. 再运行中等规模测试")
        print("3. 最后运行大规模测试")
    else:
        print("❌ 部分测试失败，请检查环境配置。")
        print("\n建议：")
        print("1. 确保C++扩展正确编译")
        print("2. 检查依赖包是否正确安装")
        print("3. 确保有足够的内存")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
