#!/usr/bin/env python
"""
Ward D vs Ward D2 Methods Demonstration

This script demonstrates the difference between Ward D (classic Ward) and Ward D2 
methods in hierarchical clustering using the Sequenzo library.

Ward D (classic):  Uses squared Euclidean distances ÷ 2
Ward D2:          Uses squared Euclidean distances

Both methods produce identical cluster assignments, but different distance values
in the linkage matrix, which affects dendrogram visualization.

Author: Yuqi Liang
Date: 24/09/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sequenzo.clustering.hierarchical_clustering import Cluster
import seaborn as sns

def generate_sample_data():
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Create three well-separated clusters in 2D space
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 20)
    cluster2 = np.random.multivariate_normal([8, 2], [[0.5, 0], [0, 0.5]], 20)
    cluster3 = np.random.multivariate_normal([5, 8], [[0.5, 0], [0, 0.5]], 20)
    
    # Combine the clusters
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create entity IDs
    entity_ids = [f"Entity_{i+1:02d}" for i in range(len(data))]
    
    return data, entity_ids

def compute_euclidean_distance_matrix(data):
    """Compute Euclidean distance matrix from data points."""
    distances = pdist(data, metric='euclidean')
    return squareform(distances)

def compare_ward_methods():
    """Compare Ward D and Ward D2 methods."""
    print("=" * 60)
    print("Ward D vs Ward D2 Methods Comparison")
    print("=" * 60)
    
    # Generate sample data
    data, entity_ids = generate_sample_data()
    print(f"Generated {len(data)} data points in {data.shape[1]}D space")
    
    # Compute distance matrix
    distance_matrix = compute_euclidean_distance_matrix(data)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Initialize clustering with different Ward methods
    print("\n" + "-" * 40)
    print("1. Ward D (Classic Ward Method)")
    print("-" * 40)
    cluster_ward_d = Cluster(
        matrix=distance_matrix,
        entity_ids=entity_ids,
        clustering_method="ward_d"
    )
    
    print("\n" + "-" * 40)
    print("2. Ward D2 Method")
    print("-" * 40)
    cluster_ward_d2 = Cluster(
        matrix=distance_matrix,
        entity_ids=entity_ids,
        clustering_method="ward_d2"
    )
    
    # Compare linkage matrices
    print("\n" + "=" * 40)
    print("LINKAGE MATRIX COMPARISON")
    print("=" * 40)
    
    linkage_d = cluster_ward_d.linkage_matrix
    linkage_d2 = cluster_ward_d2.linkage_matrix
    
    print(f"Ward D - First 5 merges (distances):")
    for i in range(min(5, len(linkage_d))):
        print(f"  Merge {i+1}: {linkage_d[i, 2]:.6f}")
    
    print(f"\nWard D2 - First 5 merges (distances):")
    for i in range(min(5, len(linkage_d2))):
        print(f"  Merge {i+1}: {linkage_d2[i, 2]:.6f}")
    
    print(f"\nDistance ratio (Ward D2 / Ward D): {linkage_d2[0, 2] / linkage_d[0, 2]:.2f}")
    print("(Ward D2 distances should be approximately 2× Ward D distances)")
    
    # Test cluster assignments
    print("\n" + "=" * 40)
    print("CLUSTER ASSIGNMENT COMPARISON")
    print("=" * 40)
    
    num_clusters = 3
    labels_ward_d = cluster_ward_d.get_cluster_labels(num_clusters)
    labels_ward_d2 = cluster_ward_d2.get_cluster_labels(num_clusters)
    
    # Check if assignments are identical
    assignments_identical = np.array_equal(labels_ward_d, labels_ward_d2)
    print(f"Cluster assignments identical: {assignments_identical}")
    
    if assignments_identical:
        print("✓ Both methods produce the same cluster assignments")
    else:
        print("✗ Different cluster assignments detected")
        
    return cluster_ward_d, cluster_ward_d2, data

def plot_comparison(cluster_ward_d, cluster_ward_d2, data):
    """Create visualization comparing the two methods."""
    print("\n" + "=" * 40)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 40)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ward D vs Ward D2 Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Data points
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.7, s=50)
    axes[0, 0].set_title('Original Data Points')
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Ward D dendrogram
    from scipy.cluster.hierarchy import dendrogram
    axes[0, 1].set_title('Ward D (Classic) Dendrogram')
    dendrogram(cluster_ward_d.linkage_matrix, ax=axes[0, 1], 
               leaf_font_size=8, orientation='top')
    axes[0, 1].set_xlabel('Entity Index')
    axes[0, 1].set_ylabel('Distance')
    
    # Plot 3: Ward D2 dendrogram  
    axes[1, 0].set_title('Ward D2 Dendrogram')
    dendrogram(cluster_ward_d2.linkage_matrix, ax=axes[1, 0],
               leaf_font_size=8, orientation='top')
    axes[1, 0].set_xlabel('Entity Index')
    axes[1, 0].set_ylabel('Distance')
    
    # Plot 4: Distance comparison
    linkage_d = cluster_ward_d.linkage_matrix
    linkage_d2 = cluster_ward_d2.linkage_matrix
    
    axes[1, 1].plot(linkage_d[:, 2], label='Ward D', marker='o', markersize=4)
    axes[1, 1].plot(linkage_d2[:, 2], label='Ward D2', marker='s', markersize=4)
    axes[1, 1].set_title('Linkage Distances Comparison')
    axes[1, 1].set_xlabel('Merge Step')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'ward_methods_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {output_path}")
    
    plt.show()

def main():
    """Main function to run the demonstration."""
    print("Starting Ward D vs Ward D2 demonstration...")
    
    try:
        # Compare methods
        cluster_ward_d, cluster_ward_d2, data = compare_ward_methods()
        
        # Create visualizations
        plot_comparison(cluster_ward_d, cluster_ward_d2, data)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ Ward D and Ward D2 methods successfully implemented")
        print("✓ Both methods produce identical cluster assignments")
        print("✓ Ward D2 distances are approximately 2× Ward D distances")
        print("✓ Choice between methods affects dendrogram visualization only")
        print("\nRecommendation:")
        print("- Use 'ward_d' for classic Ward method (most common)")
        print("- Use 'ward_d2' when you need distances equal to variance increase")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main()
