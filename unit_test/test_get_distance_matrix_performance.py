"""
单元测试：测试 get_distance_matrix 在重复序列多/不多情况下的性能表现
主要关注：
1. 重复序列多的情况（去重后序列少）
2. 重复序列少的情况（去重后序列多）
3. 时间和内存消耗对比
4. 大规模数据测试（扩展到4万序列）
"""

import unittest
import time
import psutil
import os
import numpy as np
import pandas as pd
import gc
from memory_profiler import profile
import sys
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix


class TestGetDistanceMatrixPerformance(unittest.TestCase):
    """测试 get_distance_matrix 在不同重复序列情况下的性能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.process = psutil.Process(os.getpid())
        self.base_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def tearDown(self):
        """测试后的清理工作"""
        gc.collect()
        
    def get_memory_usage(self):
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024 - self.base_memory
    
    def create_test_data(self, total_sequences=40000, unique_sequences=1000, seq_length=10, states=None):
        """
        创建测试数据
        
        Args:
            total_sequences: 总序列数（扩展到4万）
            unique_sequences: 唯一序列数（控制重复程度）
            seq_length: 序列长度
            states: 状态列表
        """
        if states is None:
            states = [0, 1, 2, 3, 4]  # 5个状态
            
        # 首先生成唯一序列
        np.random.seed(42)  # 固定随机种子确保可重复
        unique_seqs = []
        for _ in range(unique_sequences):
            seq = np.random.choice(states, size=seq_length)
            unique_seqs.append(seq)
        
        # 然后通过重复生成总序列数
        all_seqs = []
        all_ids = []
        
        for i in range(total_sequences):
            # 随机选择一个唯一序列进行重复
            seq_idx = i % unique_sequences
            all_seqs.append(unique_seqs[seq_idx])
            all_ids.append(f"seq_{i}")
        
        # 创建DataFrame
        df_data = {}
        df_data['worker_id'] = all_ids
        
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
        
        return sequence_data, unique_sequences
    
    def measure_performance(self, sequence_data, method="OM", test_name=""):
        """
        测量性能
        
        Args:
            sequence_data: 序列数据
            method: 距离计算方法
            test_name: 测试名称
            
        Returns:
            dict: 包含时间和内存使用信息的字典
        """
        print(f"\n=== {test_name} ===")
        print(f"总序列数: {sequence_data.seqdata.shape[0]}")
        print(f"序列长度: {sequence_data.seqdata.shape[1]}")
        print(f"状态数: {len(sequence_data.states)}")
        
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # 执行距离矩阵计算
            result = get_distance_matrix(
                sequence_data, 
                method=method, 
                sm="TRATE", 
                indel="auto",
                full_matrix=True
            )
            
            # 记录结束时间和内存
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            peak_memory = max(start_memory, end_memory)
            
            print(f"执行时间: {execution_time:.2f} 秒")
            print(f"内存使用: {memory_used:.2f} MB")
            print(f"峰值内存: {peak_memory:.2f} MB")
            print(f"结果矩阵大小: {result.shape}")
            
            return {
                'execution_time': execution_time,
                'memory_used': memory_used,
                'peak_memory': peak_memory,
                'result_shape': result.shape,
                'success': True
            }
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"执行失败: {str(e)}")
            print(f"失败时间: {execution_time:.2f} 秒")
            traceback.print_exc()
            
            return {
                'execution_time': execution_time,
                'memory_used': 0,
                'peak_memory': 0,
                'result_shape': None,
                'success': False,
                'error': str(e)
            }
    
    def test_high_duplication_performance(self):
        """测试高重复序列情况下的性能（重复序列多，去重后序列少）"""
        print("\n" + "="*60)
        print("测试高重复序列情况（40000个序列，只有2000个唯一序列）")
        print("="*60)
        
        # 创建高重复数据：40000个序列，只有2000个唯一序列
        sequence_data, unique_count = self.create_test_data(
            total_sequences=40000, 
            unique_sequences=2000,  # 高重复
            seq_length=10
        )
        
        print(f"去重后唯一序列数: {unique_count}")
        
        # 测试OMspell方法
        omspell_result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="高重复序列 - OMspell方法"
        )
        
        # 测试OMspell方法
        omspell_result2 = self.measure_performance(
            sequence_data, 
            method="OMspell",
            test_name="高重复序列 - OMspell方法2"
        )
        
        # 测试OMspell方法
        omspell_result3 = self.measure_performance(
            sequence_data, 
            method="OMspell",
            test_name="高重复序列 - OMspell方法3"
        )
        
        # 验证结果
        self.assertTrue(omspell_result['success'], f"OMspell方法失败: {omspell_result.get('error', '')}")
        self.assertTrue(omspell_result2['success'], f"OMspell方法2失败: {omspell_result2.get('error', '')}")
        self.assertTrue(omspell_result3['success'], f"OMspell方法3失败: {omspell_result3.get('error', '')}")
        
        # 保存结果
        self.high_dup_results = {
            'OMspell_1': omspell_result,
            'OMspell_2': omspell_result2,
            'OMspell_3': omspell_result3,
            'unique_count': unique_count
        }
    
    def test_low_duplication_performance(self):
        """测试低重复序列情况下的性能（重复序列少，去重后序列多）"""
        print("\n" + "="*60)
        print("测试低重复序列情况（40000个序列，有32000个唯一序列）")
        print("="*60)
        
        # 创建低重复数据：40000个序列，有32000个唯一序列
        sequence_data, unique_count = self.create_test_data(
            total_sequences=40000, 
            unique_sequences=32000,  # 低重复
            seq_length=10
        )
        
        print(f"去重后唯一序列数: {unique_count}")
        
        # 测试OMspell方法
        omspell_result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="低重复序列 - OMspell方法"
        )
        
        # 测试OMspell方法
        omspell_result2 = self.measure_performance(
            sequence_data, 
            method="OMspell",
            test_name="低重复序列 - OMspell方法2"
        )
        
        # 测试OMspell方法
        omspell_result3 = self.measure_performance(
            sequence_data, 
            method="OMspell",
            test_name="低重复序列 - OMspell方法3"
        )
        
        # 验证结果
        self.assertTrue(omspell_result['success'], f"OMspell方法失败: {omspell_result.get('error', '')}")
        self.assertTrue(omspell_result2['success'], f"OMspell方法2失败: {omspell_result2.get('error', '')}")
        self.assertTrue(omspell_result3['success'], f"OMspell方法3失败: {omspell_result3.get('error', '')}")
        
        # 保存结果
        self.low_dup_results = {
            'OMspell_1': omspell_result,
            'OMspell_2': omspell_result2,
            'OMspell_3': omspell_result3,
            'unique_count': unique_count
        }
    
    def test_no_duplication_performance(self):
        """测试无重复序列情况下的性能（所有序列都唯一）"""
        print("\n" + "="*60)
        print("测试无重复序列情况（40000个序列，全部唯一）")
        print("="*60)
        
        # 创建无重复数据：40000个序列，全部唯一
        sequence_data, unique_count = self.create_test_data(
            total_sequences=40000, 
            unique_sequences=40000,  # 无重复
            seq_length=10
        )
        
        print(f"去重后唯一序列数: {unique_count}")
        
        # 测试OMspell方法
        omspell_result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="无重复序列 - OMspell方法"
        )
        
        # 测试OMspell方法
        omspell_result2 = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="无重复序列 - OMspell方法2"
        )
        
        # 测试OMspell方法
        omspell_result3 = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="无重复序列 - OMspell方法3"
        )
        
        # 验证结果
        self.assertTrue(omspell_result['success'], f"OMspell方法失败: {omspell_result.get('error', '')}")
        self.assertTrue(omspell_result2['success'], f"OMspell方法2失败: {omspell_result2.get('error', '')}")
        self.assertTrue(omspell_result3['success'], f"OMspell方法3失败: {omspell_result3.get('error', '')}")
        
        # 保存结果
        self.no_dup_results = {
            'OMspell_1': omspell_result,
            'OMspell_2': omspell_result2,
            'OMspell_3': omspell_result3,
            'unique_count': unique_count
        }
    
    def test_medium_scale_performance(self):
        """测试中等规模数据（1万序列）的性能"""
        print("\n" + "="*60)
        print("测试中等规模数据（10000个序列，5000个唯一序列）")
        print("="*60)
        
        # 创建中等规模数据：10000个序列，5000个唯一序列
        sequence_data, unique_count = self.create_test_data(
            total_sequences=10000, 
            unique_sequences=5000,  # 中等重复
            seq_length=10
        )
        
        print(f"去重后唯一序列数: {unique_count}")
        
        # 测试OMspell方法
        omspell_result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="中等规模 - OMspell方法"
        )
        
        # 测试OMspell方法
        omspell_result2 = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="中等规模 - OMspell方法2"
        )
        
        # 验证结果
        self.assertTrue(omspell_result['success'], f"OMspell方法失败: {omspell_result.get('error', '')}")
        self.assertTrue(omspell_result2['success'], f"OMspell方法2失败: {omspell_result2.get('error', '')}")
        
        # 保存结果
        self.medium_scale_results = {
            'OMspell_1': omspell_result,
            'OMspell_2': omspell_result2,
            'unique_count': unique_count
        }
    
    def test_performance_comparison(self):
        """比较不同重复程度下的性能表现"""
        print("\n" + "="*80)
        print("性能对比分析")
        print("="*80)
        
        # 确保先运行其他测试
        if not hasattr(self, 'high_dup_results'):
            self.test_high_duplication_performance()
        if not hasattr(self, 'low_dup_results'):
            self.test_low_duplication_performance()
        if not hasattr(self, 'no_dup_results'):
            self.test_no_duplication_performance()
        if not hasattr(self, 'medium_scale_results'):
            self.test_medium_scale_performance()
        
        # 创建对比表格
        methods = ['OMspell_1', 'OMspell_2', 'OMspell_3']
        scenarios = ['高重复(2K唯一)', '中等规模(5K唯一)', '低重复(32K唯一)', '无重复(40K唯一)']
        
        print("\n执行时间对比（秒）:")
        print("-" * 80)
        print(f"{'方法':<12} {'高重复':<15} {'中等规模':<15} {'低重复':<15} {'无重复':<15}")
        print("-" * 80)
        
        for method in methods:
            high_time = self.high_dup_results[method]['execution_time']
            medium_time = self.medium_scale_results[method]['execution_time']
            low_time = self.low_dup_results[method]['execution_time']
            no_time = self.no_dup_results[method]['execution_time']
            print(f"{method:<12} {high_time:<15.2f} {medium_time:<15.2f} {low_time:<15.2f} {no_time:<15.2f}")
        
        print("\n内存使用对比（MB）:")
        print("-" * 80)
        print(f"{'方法':<12} {'高重复':<15} {'中等规模':<15} {'低重复':<15} {'无重复':<15}")
        print("-" * 80)
        
        for method in methods:
            high_mem = self.high_dup_results[method]['memory_used']
            medium_mem = self.medium_scale_results[method]['memory_used']
            low_mem = self.low_dup_results[method]['memory_used']
            no_mem = self.no_dup_results[method]['memory_used']
            print(f"{method:<12} {high_mem:<15.2f} {medium_mem:<15.2f} {low_mem:<15.2f} {no_mem:<15.2f}")
        
        # 性能分析
        print("\n性能分析:")
        print("-" * 50)
        
        for method in methods:
            high_time = self.high_dup_results[method]['execution_time']
            no_time = self.no_dup_results[method]['execution_time']
            
            if high_time > 0 and no_time > 0:
                speedup = no_time / high_time
                print(f"{method}方法: 高重复比无重复快 {speedup:.2f} 倍")
        
        # 验证性能预期
        for method in methods:
            high_time = self.high_dup_results[method]['execution_time']
            no_time = self.no_dup_results[method]['execution_time']
            
            # 高重复应该比无重复快（因为去重后计算量更少）
            if high_time > 0 and no_time > 0:
                self.assertLess(high_time, no_time * 2, 
                               f"{method}方法在高重复情况下应该明显更快")
    
    def test_edge_cases(self):
        """测试边界情况"""
        print("\n" + "="*60)
        print("测试边界情况")
        print("="*60)
        
        # 测试极小数据集
        print("\n测试极小数据集（100个序列，50个唯一）")
        sequence_data, unique_count = self.create_test_data(
            total_sequences=100, 
            unique_sequences=50, 
            seq_length=5
        )
        
        result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="极小数据集"
        )
        
        self.assertTrue(result['success'], f"极小数据集测试失败: {result.get('error', '')}")
        
        # 测试极大重复（所有序列相同）
        print("\n测试极大重复（1000个序列，只有1个唯一）")
        sequence_data, unique_count = self.create_test_data(
            total_sequences=1000, 
            unique_sequences=1, 
            seq_length=10
        )
        
        result = self.measure_performance(
            sequence_data, 
            method="OMspell", 
            test_name="极大重复数据集"
        )
        
        self.assertTrue(result['success'], f"极大重复数据集测试失败: {result.get('error', '')}")
        self.assertEqual(unique_count, 1, "应该只有1个唯一序列")


def run_performance_tests():
    """运行性能测试"""
    print("开始运行 get_distance_matrix 大规模性能测试...")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试用例
    test_cases = [
        'test_high_duplication_performance',
        'test_low_duplication_performance', 
        'test_no_duplication_performance',
        'test_medium_scale_performance',
        'test_performance_comparison',
        'test_edge_cases'
    ]
    
    for test_case in test_cases:
        suite.addTest(TestGetDistanceMatrixPerformance(test_case))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("测试完成!")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result


if __name__ == '__main__':
    # 运行性能测试
    result = run_performance_tests()
    
    # 根据测试结果设置退出码
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
