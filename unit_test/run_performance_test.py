#!/usr/bin/env python3
"""
运行 get_distance_matrix 性能测试的脚本
可以选择运行完整版或简化版测试
"""

import sys
import os

def main():
    print("get_distance_matrix 大规模性能测试")
    print("="*60)
    print("1. 完整版测试（需要 psutil 和 memory_profiler）")
    print("2. 简化版测试（只需要标准库）")
    print("3. 退出")
    print("\n注意：测试规模已扩展到4万序列，请确保有足够的内存和计算资源")
    
    while True:
        try:
            choice = input("\n请选择测试类型 (1-3): ").strip()
            
            if choice == '1':
                print("\n尝试运行完整版测试...")
                try:
                    from test_get_distance_matrix_performance import run_performance_tests
                    result = run_performance_tests()
                    break
                except ImportError as e:
                    print(f"导入错误: {e}")
                    print("请安装所需依赖: pip install psutil memory_profiler")
                    print("或者选择简化版测试")
                    continue
                    
            elif choice == '2':
                print("\n运行简化版测试...")
                from test_get_distance_matrix_simple import run_simple_performance_tests
                result = run_simple_performance_tests()
                break
                
            elif choice == '3':
                print("退出测试")
                return 0
                
            else:
                print("无效选择，请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n测试被用户中断")
            return 1
        except Exception as e:
            print(f"发生错误: {e}")
            continue
    
    # 返回测试结果
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

