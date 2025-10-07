#!/usr/bin/env python
"""
快速验证 Sequenzo 安装是否正常

运行方式:
    python verify_installation.py
    
或者直接执行:
    ./verify_installation.py
"""

def verify_installation():
    """验证 Sequenzo 是否正确安装并能正常工作"""
    
    print("="*60)
    print("Sequenzo 安装验证")
    print("="*60)
    
    # 1. 测试基本导入
    print("\n[1/7] 测试基本导入...")
    try:
        import sequenzo
        print(f"    OK - Sequenzo 版本: {sequenzo.__version__}")
    except Exception as e:
        print(f"    [X] 导入失败: {e}")
        return False
    
    # 2. 测试 C++ 扩展
    print("\n[2/7] 测试 C++ 扩展...")
    try:
        import sequenzo.clustering.clustering_c_code as cc
        print("    OK - 聚类 C++ 扩展加载成功")
    except Exception as e:
        print(f"    [!] 聚类 C++ 扩展加载失败: {e}")
    
    try:
        from sequenzo.dissimilarity_measures import c_code
        print("    OK - 距离计算 C++ 扩展加载成功")
    except Exception as e:
        print(f"    [!] 距离计算 C++ 扩展加载失败: {e}")
    
    # 3. 测试数据加载
    print("\n[3/7] 测试数据加载...")
    try:
        from sequenzo import list_datasets, load_dataset
        datasets = list_datasets()
        print(f"    OK - 找到 {len(datasets)} 个数据集")
        
        df = load_dataset('country_co2_emissions_global_deciles')
        print(f"    OK - 成功加载测试数据集 (形状: {df.shape})")
    except Exception as e:
        print(f"    [X] 数据加载失败: {e}")
        return False
    
    # 4. 测试 SequenceData 创建
    print("\n[4/7] 测试序列数据对象创建...")
    try:
        from sequenzo import SequenceData
        time_list = list(df.columns)[1:]
        states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
        
        sequence_data = SequenceData(
            df, 
            time=time_list, 
            id_col="country", 
            states=states,
            labels=states
        )
        print(f"    OK - 创建成功 ({sequence_data.num_sequences} 个序列, {sequence_data.num_time_points} 个时间点)")
    except Exception as e:
        print(f"    [X] SequenceData 创建失败: {e}")
        return False
    
    # 5. 测试可视化（不保存文件）
    print("\n[5/7] 测试可视化功能...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        from sequenzo import plot_sequence_index, plot_state_distribution
        
        # 只是运行函数，不保存文件
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_sequence_index(sequence_data)
            plot_state_distribution(sequence_data)
        
        print("    OK - 可视化功能正常")
    except Exception as e:
        print(f"    [!] 可视化测试失败: {e}")
    
    # 6. 测试距离矩阵计算
    print("\n[6/7] 测试距离矩阵计算...")
    try:
        from sequenzo import get_distance_matrix
        
        om = get_distance_matrix(
            seqdata=sequence_data,
            method="OM",
            sm="TRATE",
            indel="auto"
        )
        print(f"    OK - 距离矩阵计算成功 (形状: {om.shape})")
    except Exception as e:
        print(f"    [X] 距离矩阵计算失败: {e}")
        return False
    
    # 7. 测试聚类分析
    print("\n[7/7] 测试聚类分析...")
    try:
        from sequenzo import Cluster, ClusterQuality, ClusterResults
        
        cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
        print("    OK - 聚类对象创建成功")
        
        cluster_quality = ClusterQuality(cluster)
        cluster_quality.compute_cluster_quality_scores()
        summary = cluster_quality.get_cqi_table()
        print(f"    OK - 聚类质量评估完成 ({len(summary)} 个指标)")
        
        cluster_results = ClusterResults(cluster)
        membership = cluster_results.get_cluster_memberships(num_clusters=5)
        print(f"    OK - 聚类成员提取成功 ({len(membership)} 个序列)")
    except Exception as e:
        print(f"    [X] 聚类分析失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("OK - 所有核心功能验证通过！")
    print("  Sequenzo 已正确安装并可以正常使用。")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import sys
    success = verify_installation()
    sys.exit(0 if success else 1)

