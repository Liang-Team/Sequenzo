#!/usr/bin/env python3
"""
Export distance matrix to compare with R exactly
"""

from sequenzo import *
import pandas as pd
import numpy as np

def export_for_r_comparison():
    # Load data exactly as R does
    df = load_dataset('mvad')
    
    # Only use the time columns that R uses (15:86 in R, which is 0-based indexing)
    # R uses columns 15-86, which in Python DataFrame would be index 14-85
    time_cols = df.columns[14:86].tolist()  # This should match R's 15:86
    
    print("Time columns used:")
    print(f"Number of time columns: {len(time_cols)}")
    print(f"First 5: {time_cols[:5]}")
    print(f"Last 5: {time_cols[-5:]}")
    
    # Get unique states from the time columns
    unique_states = set()
    for col in time_cols:
        unique_states.update(df[col].unique())
    unique_states = sorted(list(unique_states))
    
    print(f"\nUnique states in data: {unique_states}")
    
    # Create SequenceData with actual states found in data
    states = ['FE', 'HE', 'employment', 'joblessness', 'school', 'training']
    labels = ['further education', 'higher education', 'employment', 'joblessness', 'school', 'training']
    
    sequence_data = SequenceData(df, 
                                 time=time_cols, 
                                 id_col='id', 
                                 states=states,
                                 labels=labels)
    
    # Compute OM with exact R parameters
    om = get_distance_matrix(sequence_data, 
                            method='OM', 
                            sm='CONSTANT',  # Same as R
                            indel=1)       # Same as R
    
    # Save the distance matrix for comparison
    om_array = om.values if hasattr(om, 'values') else om
    np.savetxt('/Users/lei/Documents/Sequenzo_all_folders/Sequenzo-main/python_distance_matrix.csv', 
               om_array, delimiter=',', fmt='%.6f')
    
    print(f"\nDistance matrix exported to python_distance_matrix.csv")
    print(f"Shape: {om_array.shape}")
    print(f"Min: {np.min(om_array):.6f}, Max: {np.max(om_array):.6f}")
    print(f"Mean: {np.mean(om_array):.6f}")
    
    # Test clustering
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward')
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    
    print(f"\nClustering results:")
    summary_table = cluster_quality.get_metrics_table()
    print(summary_table)

if __name__ == "__main__":
    export_for_r_comparison()
