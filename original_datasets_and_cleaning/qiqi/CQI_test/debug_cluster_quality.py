#!/usr/bin/env python3
"""
Debug script to compare clustering quality results with R WeightedCluster package
"""

from sequenzo import *
import pandas as pd
import numpy as np

def main():
    # Load the MVAD dataset
    df = load_dataset('mvad')

    # Define time columns (same as R)
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

    # Use correct states
    states = ['FE', 'HE', 'employment', 'joblessness', 'school', 'training']
    labels = ['further education', 'higher education', 'employment', 'joblessness', 'school', 'training']

    # Create sequence data
    sequence_data = SequenceData(df, 
                                 time=time_list, 
                                 id_col='id', 
                                 states=states,
                                 labels=labels)

    # Compute OM distance matrix (same parameters as R)
    om = get_distance_matrix(sequence_data, 
                            method='OM', 
                            sm='CONSTANT', 
                            indel=1)

    # Create cluster with ward method
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward')

    # Compute cluster quality scores
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()

    # Print detailed results for comparison
    print("DETAILED COMPARISON WITH R RESULTS")
    print("=" * 50)
    
    print("\nPython results (detailed):")
    for k in range(2, 21):
        if k-2 < len(cluster_quality.scores['PBC']):
            pbc = cluster_quality.scores['PBC'][k-2]
            asw = cluster_quality.scores['ASW'][k-2]
            ch = cluster_quality.scores['CH'][k-2]
            r2 = cluster_quality.scores['R2'][k-2]
            hc = cluster_quality.scores['HC'][k-2]
            hg = cluster_quality.scores['HG'][k-2]
            
            print(f"k={k:2d}: PBC={pbc:.6f}, ASW={asw:.6f}, CH={ch:.6f}, R2={r2:.6f}, HC={hc:.6f}, HG={hg:.6f}")

    print("\nR results for comparison:")
    print("k= 2: PBC=0.425100, ASW=0.425100, CH=209.816084, R2=0.710832, HC=0.065480, HG=0.907965")
    print("k= 6: PBC=0.668392, ASW=0.379087, CH=..., R2=..., HC=..., HG=...")
    print("k=20: PBC=..., ASW=..., CH=..., R2=0.710832, HC=0.065480, HG=0.907965")
    
    # Show optimal clusters comparison
    summary_table = cluster_quality.get_metrics_table()
    print("\nOptimal clusters comparison:")
    print(summary_table)

if __name__ == "__main__":
    main()
