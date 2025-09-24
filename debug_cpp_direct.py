#!/usr/bin/env python3
"""
Direct test of C++ cluster quality functions to debug differences with R
"""

from sequenzo import *
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster

def test_direct_cpp():
    # Load the MVAD dataset
    df = load_dataset('mvad')

    # Define time columns
    time_list = ['Jul.93', 'Aug.93', 'Sep.93', 'Oct.93', 'Nov.93', 'Dec.93',
           'Jan.94', 'Feb.94', 'Mar.94', 'Apr.94', 'May.94', 'Jun.94', 'Jul.94',
           'Aug.94', 'Sep.94', 'Oct.94', 'Nov.94', 'Dec.94', 'Jan.95', 'Feb.95',
           'Mar.95', 'Apr.95', 'May.95', 'Jun.95', 'Jul.95', 'Aug.95', 'Sep.95',
           'Oct.95', 'Nov.95', 'Dec.95', 'Jan.96', 'Feb.96', 'Mar.96', 'Apr.96',
           'May.96', 'Jun.96', 'Jul.96', 'Aug.96', 'Sep.96', 'Oct.96', 'Nov.96',
           'Dec.96', 'Jan.97', 'Feb.97', 'Mar.97', 'Apr.97', 'May.97', 'Jun.97',
           'Jul.97', 'Aug.97', 'Sep.97', 'Oct.97', 'Nov.97', 'Dec.97', 'Jan.98',
           'Feb.98', 'Mar.98', 'Apr.98', 'May.98', 'Jun.98', 'Jul.98', 'Aug.98',
           'Sep.98', 'Oct.98', 'Nov.98', 'Dec.98', 'Jan.99', 'Feb.99', 'Mar.99',
           'Apr.99', 'May.99', 'Jun.99']

    states = ['FE', 'HE', 'employment', 'joblessness', 'school', 'training']
    labels = ['further education', 'higher education', 'employment', 'joblessness', 'school', 'training']

    # Create sequence data
    sequence_data = SequenceData(df, 
                                 time=time_list, 
                                 id_col='id', 
                                 states=states,
                                 labels=labels)

    # Compute OM distance matrix
    om = get_distance_matrix(sequence_data, 
                            method='OM', 
                            sm='CONSTANT', 
                            indel=1)

    # Create cluster with ward method
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward')

    # Test direct C++ calls for specific k values
    from sequenzo.clustering import clustering_c_code
    from scipy.spatial.distance import squareform
    
    print("DIRECT C++ RESULTS COMPARISON")
    print("=" * 50)
    
    # Test k=2, k=6, k=20 specifically to compare with R
    test_ks = [2, 6, 7, 20]
    condensed = squareform(cluster.full_matrix)
    weights = cluster.weights
    
    for k in test_ks:
        labels = fcluster(cluster.linkage_matrix, k, criterion="maxclust")
        
        # Call C++ function directly
        result = clustering_c_code.cluster_quality_condensed(
            condensed.astype(np.float64, copy=False),
            labels.astype(np.int32, copy=False),
            weights.astype(np.float64, copy=False),
            cluster.full_matrix.shape[0],
            k
        )
        
        print(f"\nk={k}:")
        print(f"  PBC: {result.get('PBC', 'N/A'):.6f}")
        print(f"  ASW: {result.get('ASW', 'N/A'):.6f}")
        print(f"  CH:  {result.get('CH', 'N/A'):.6f}")
        print(f"  R2:  {result.get('R2', 'N/A'):.6f}")
        print(f"  HC:  {result.get('HC', 'N/A'):.6f}")
        print(f"  HG:  {result.get('HG', 'N/A'):.6f}")
        print(f"  CHsq:{result.get('CHsq', 'N/A'):.6f}")
        print(f"  R2sq:{result.get('R2sq', 'N/A'):.6f}")
        
        # Show all available keys
        print(f"  Available keys: {list(result.keys())}")

    print("\nR REFERENCE VALUES:")
    print("k=2:  PBC=0.425100, ASW=0.425100, CH=209.816084, R2=0.710832, HC=0.065480")
    print("k=6:  PBC=0.668392, ASW=0.379087")
    print("k=20: R2=0.710832, HC=0.065480, HG=0.907965")

if __name__ == "__main__":
    test_direct_cpp()
