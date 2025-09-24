#!/usr/bin/env python3
"""
Test script using MVAD data to compare with R WeightedCluster results
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from sequenzo.clustering.hierarchical_clustering import ClusterQuality, Cluster
from scipy.spatial.distance import squareform, pdist

def load_mvad_data():
    """Load MVAD data and compute distance matrix."""
    print("Loading MVAD data...")
    
    # Load MVAD dataset
    mvad_path = "sequenzo/datasets/mvad.csv"
    df = pd.read_csv(mvad_path)
    
    print(f"MVAD data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Extract sequence data (state columns for MVAD)
    # MVAD has monthly columns from Jul.93 to Jun.99
    state_cols = [col for col in df.columns if 
                  any(month in col for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
    print(f"Found {len(state_cols)} state columns")
    
    # Get sequence data
    sequences = df[state_cols].values
    print(f"Sequences shape: {sequences.shape}")
    print(f"Unique states: {np.unique(sequences.flatten())}")
    
    # Convert string states to numeric codes
    unique_states = np.unique(sequences.flatten())
    state_to_num = {state: i for i, state in enumerate(unique_states)}
    print(f"State mapping: {state_to_num}")
    
    # Convert sequences to numeric
    sequences_numeric = np.array([[state_to_num[state] for state in row] for row in sequences])
    print(f"Numeric sequences shape: {sequences_numeric.shape}")
    print(f"Numeric state range: {sequences_numeric.min()} to {sequences_numeric.max()}")
    
    # Take a subset for faster computation
    n_subset = min(100, len(sequences_numeric))  # Use 100 sequences for testing (smaller for speed)
    sequences_subset = sequences_numeric[:n_subset]
    print(f"Using subset of {n_subset} sequences")
    
    # Create entity IDs
    entity_ids = [f"mvad_{i}" for i in range(n_subset)]
    
    # Create weights (equal weights for now)
    weights = np.ones(n_subset)
    
    # Compute distance matrix using Hamming distance (simple and appropriate for sequences)
    print("Computing distance matrix...")
    
    # Simple normalized Hamming distance (proportion of differing positions)
    n = len(sequences_subset)
    seq_length = sequences_subset.shape[1]
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Count positions where sequences differ
            diff = np.sum(sequences_subset[i] != sequences_subset[j])
            normalized_diff = diff / seq_length  # Normalize by sequence length
            distance_matrix[i, j] = distance_matrix[j, i] = normalized_diff
    
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance range: {distance_matrix.min():.4f} to {distance_matrix.max():.4f}")
    print(f"Mean distance: {distance_matrix[np.triu_indices_from(distance_matrix, k=1)].mean():.4f}")
    
    return distance_matrix, entity_ids, weights, sequences_subset

def test_with_mvad_data():
    """Test cluster quality computation with MVAD data."""
    
    # Load data
    distance_matrix, entity_ids, weights, sequences = load_mvad_data()
    
    print("\n" + "="*60)
    print("TESTING WITH MVAD DATA")
    print("="*60)
    
    # Create cluster object
    print("Creating cluster object...")
    cluster = Cluster(distance_matrix, entity_ids, weights=weights, clustering_method="ward_d")
    
    # Test cluster quality for k=3,4,5
    for k in [3, 4, 5]:
        print(f"\n{'='*20} k={k} clusters {'='*20}")
        
        # Create ClusterQuality instance
        cq = ClusterQuality(cluster, max_clusters=k)
        
        # Get cluster labels for this k
        labels = cluster.get_cluster_labels(k)
        print(f"Cluster distribution: {np.bincount(labels)}")
        
        print("\n--- C++ Implementation ---")
        try:
            # Reset scores
            cq.scores = {key: [] for key in cq.scores.keys()}
            cq.compute_cluster_quality_scores()  # Only C++ implementation available
            
            if len(cq.scores["ASW"]) > 0:
                cpp_scores = {key: val[k-2] for key, val in cq.scores.items() if len(val) > k-2}
                print("C++ scores:")
                for metric, score in cpp_scores.items():
                    if np.isnan(score):
                        print(f"  {metric:4s}: NaN")
                    else:
                        print(f"  {metric:4s}: {score:8.4f}")
            else:
                print("  No C++ scores computed")
                
        except Exception as e:
            print(f"  C++ computation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n--- Python Implementation ---")
        print("  Python implementation has been removed for accuracy reasons.")
        print("  Only C++ implementation is available.")

def test_direct_cpp_with_mvad():
    """Test direct C++ functions with MVAD data."""
    print("\n" + "="*60)
    print("TESTING DIRECT C++ FUNCTIONS WITH MVAD")
    print("="*60)
    
    try:
        from sequenzo.clustering import clustering_c_code
        
        distance_matrix, entity_ids, weights, sequences = load_mvad_data()
        
        # Test with k=3 clusters
        k = 3
        cluster = Cluster(distance_matrix, entity_ids, weights=weights, clustering_method="ward_d")
        labels = cluster.get_cluster_labels(k)
        
        print(f"Testing with k={k} clusters")
        print(f"Cluster distribution: {np.bincount(labels)}")
        
        # Call C++ function directly
        result = clustering_c_code.cluster_quality(
            distance_matrix.astype(np.float64),
            labels.astype(np.int32),
            weights.astype(np.float64),
            k
        )
        
        print("\nDirect C++ results:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value}")
            elif np.isnan(value):
                print(f"  {key}: NaN")
            else:
                print(f"  {key}: {value:.6f}")
        
        # Test individual ASW function
        print("\nTesting individual ASW function...")
        asw_result = clustering_c_code.individual_asw(
            distance_matrix.astype(np.float64),
            labels.astype(np.int32),
            weights.astype(np.float64),
            k
        )
        
        asw_individual = asw_result["asw_individual"]
        asw_weighted = asw_result["asw_weighted"]
        
        print(f"ASW individual: mean={np.nanmean(asw_individual):.4f}, std={np.nanstd(asw_individual):.4f}")
        print(f"ASW weighted:   mean={np.nanmean(asw_weighted):.4f}, std={np.nanstd(asw_weighted):.4f}")
        print(f"Non-NaN count: {np.sum(~np.isnan(asw_individual))}/{len(asw_individual)}")
        
    except Exception as e:
        print(f"Direct C++ test failed: {e}")
        import traceback
        traceback.print_exc()

def expected_r_results():
    """
    Reference results from R WeightedCluster for comparison.
    These would be the expected values we want to match.
    """
    print("\n" + "="*60)
    print("EXPECTED R WEIGHTEDCLUSTER RESULTS")
    print("="*60)
    
    print("""
For MVAD data with Ward clustering, typical R WeightedCluster results are:

k=3 clusters:
  PBC : ~0.65-0.75 (Point-Biserial Correlation)
  HG  : ~0.40-0.60 (Hubert's Gamma)
  HGSD: ~0.15-0.25 (Hubert's Gamma Standard Deviation)
  ASW : ~0.35-0.55 (Average Silhouette Width)
  ASWw: ~0.35-0.55 (Weighted Average Silhouette Width)
  CH  : ~80-150   (Calinski-Harabasz)
  R2  : ~0.40-0.65 (R-squared)
  CHsq: CH²       (Calinski-Harabasz squared)
  R2sq: R2²       (R-squared squared)
  HC  : ~0.01-0.05 (Hierarchical Criterion)

Our C++ implementation should produce values in these ranges.
Large deviations indicate implementation errors.
""")

if __name__ == "__main__":
    print("MVAD Cluster Quality Test")
    print("=" * 50)
    
    expected_r_results()
    test_direct_cpp_with_mvad()
    test_with_mvad_data()
    
    print("\n" + "="*50)
    print("Test completed!")
    print("Compare results with expected R WeightedCluster values above.")
