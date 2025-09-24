#!/usr/bin/env python3
"""
Debug distance matrix and clustering to see if they match R exactly
"""

from sequenzo import *
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster

def debug_matrix_and_clustering():
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

    print("DISTANCE MATRIX PROPERTIES:")
    print(f"Shape: {om.shape}")
    print(f"Min distance: {np.min(om):.6f}")
    print(f"Max distance: {np.max(om):.6f}")
    print(f"Mean distance: {np.mean(om):.6f}")
    print(f"Is symmetric: {np.allclose(om, om.T)}")
    print(f"Diagonal all zeros: {np.allclose(np.diag(om), 0)}")
    print(f"Has NaN/Inf: {np.any(np.isnan(om)) or np.any(np.isinf(om))}")
    
    # Show a sample of the distance matrix for verification
    print(f"\nSample distances (first 5x5):")
    om_array = om.values if hasattr(om, 'values') else om
    print(om_array[:5, :5])
    
    # Create cluster
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward')
    
    # Show clustering for k=2 and k=6
    for k in [2, 6]:
        labels = fcluster(cluster.linkage_matrix, k, criterion="maxclust")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nk={k} clustering:")
        print(f"  Cluster sizes: {dict(zip(unique_labels, counts))}")
        print(f"  First 10 cluster labels: {labels[:10]}")

if __name__ == "__main__":
    debug_matrix_and_clustering()
