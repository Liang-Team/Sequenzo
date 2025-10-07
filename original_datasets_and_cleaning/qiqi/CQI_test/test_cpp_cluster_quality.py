#!/usr/bin/env python3
"""
Test script for C++ cluster quality implementation
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from sequenzo.clustering.hierarchical_clustering import ClusterQuality, Cluster
from scipy.spatial.distance import pdist, squareform

def create_test_data():
    """Create simple test data for validation."""
    np.random.seed(42)
    
    # Create synthetic data with clear clusters
    cluster1 = np.random.normal([0, 0], 0.5, (20, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (20, 2))
    cluster3 = np.random.normal([0, 3], 0.5, (20, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Compute distance matrix
    distances = pdist(data, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Create entity IDs
    entity_ids = [f"seq_{i}" for i in range(len(data))]
    
    # Create weights (equal weights for now)
    weights = np.ones(len(data))
    
    return distance_matrix, entity_ids, weights

def test_cpp_vs_python():
    """Test C++ implementation vs Python implementation."""
    print("Creating test data...")
    distance_matrix, entity_ids, weights = create_test_data()
    
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Number of entities: {len(entity_ids)}")
    
    # Create cluster object
    cluster = Cluster(distance_matrix, entity_ids, weights=weights, clustering_method="ward_d")
    
    # Test both implementations for k=3 clusters
    k = 3
    print(f"\nTesting cluster quality computation for k={k}...")
    
    # Create ClusterQuality instance
    cq = ClusterQuality(cluster, max_clusters=5)
    
    print("\n=== Testing C++ Implementation ===")
    try:
        cq.compute_cluster_quality_scores(use_cpp=True)
        print("C++ implementation completed successfully!")
        
        cpp_scores = {key: val[k-2] for key, val in cq.scores.items()}  # k=3 is index 1
        print(f"C++ scores for k={k}:")
        for metric, score in cpp_scores.items():
            print(f"  {metric}: {score:.6f}")
            
    except Exception as e:
        print(f"C++ implementation failed: {e}")
        import traceback
        traceback.print_exc()
        cpp_scores = None
    
    print("\n=== Testing Python Implementation ===")
    try:
        # Reset scores for fair comparison
        cq.scores = {
            "ASW": [],
            "ASWw": [],
            "HG": [],
            "PBC": [],
            "CH": [],
            "R2": [],
            "HC": [],
        }
        
        cq.compute_cluster_quality_scores(use_cpp=False)
        print("Python implementation completed successfully!")
        
        python_scores = {key: val[k-2] for key, val in cq.scores.items()}  # k=3 is index 1
        print(f"Python scores for k={k}:")
        for metric, score in python_scores.items():
            print(f"  {metric}: {score:.6f}")
            
    except Exception as e:
        print(f"Python implementation failed: {e}")
        import traceback
        traceback.print_exc()
        python_scores = None
    
    # Compare results if both succeeded
    if cpp_scores and python_scores:
        print("\n=== Comparison ===")
        for metric in cpp_scores.keys():
            cpp_val = cpp_scores[metric]
            py_val = python_scores[metric]
            diff = abs(cpp_val - py_val) if not (np.isnan(cpp_val) or np.isnan(py_val)) else 0
            status = "OK" if diff < 0.01 else "FAIL"
            print(f"  {metric:4s}: C++={cpp_val:8.4f}, Python={py_val:8.4f}, diff={diff:8.4f} {status}")

def test_individual_cpp_functions():
    """Test individual C++ functions directly."""
    print("\n=== Testing Individual C++ Functions ===")
    
    try:
        from sequenzo.clustering import clustering_c_code
        print("Successfully imported clustering_c_code module")
        
        distance_matrix, entity_ids, weights = create_test_data()
        
        # Create simple cluster labels for testing
        n = len(entity_ids)
        labels = np.array([1]*20 + [2]*20 + [3]*20, dtype=np.int32)  # 3 clusters
        
        print(f"Testing cluster_quality function...")
        print(f"Matrix shape: {distance_matrix.shape}")
        print(f"Labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
        print(f"Weights shape: {weights.shape}")
        
        result = clustering_c_code.cluster_quality(
            distance_matrix.astype(np.float64),
            labels,
            weights.astype(np.float64),
            3  # number of clusters
        )
        
        print("C++ function result:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.6f}")
                
    except Exception as e:
        print(f"Direct C++ function test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing C++ Cluster Quality Implementation")
    print("=" * 50)
    
    test_individual_cpp_functions()
    test_cpp_vs_python()
    
    print("\nTest completed!")
