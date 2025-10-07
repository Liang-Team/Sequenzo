"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_quickstart_integration.py
@Time    : 07/10/2025 22:13
@Desc    : Integration test based on the quickstart tutorial
           Tests the complete workflow that users would typically follow
"""
import pytest
import pandas as pd
import numpy as np
from sequenzo import *


def test_dataset_loading():
    """Test that datasets can be loaded successfully"""
    # List available datasets
    datasets = list_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert 'country_co2_emissions_global_deciles' in datasets
    
    # Load a dataset
    df = load_dataset('country_co2_emissions_global_deciles')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'country' in df.columns


def test_sequence_data_creation():
    """Test SequenceData object creation"""
    df = load_dataset('country_co2_emissions_global_deciles')
    
    # Define time-span variable
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    # Create SequenceData object
    sequence_data = SequenceData(
        df, 
        time=time_list, 
        id_col="country", 
        states=states,
        labels=states
    )
    
    assert sequence_data is not None
    assert sequence_data.num_sequences > 0
    assert sequence_data.num_time_points > 0
    assert len(sequence_data.states) >= len(states)  # May include 'Missing'


def test_visualizations_no_save():
    """Test that visualization functions run without errors (without saving files)"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    
    # Test various visualization functions (matplotlib will render in memory)
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    
    # Index plot
    plot_sequence_index(sequence_data)
    
    # Legend plot
    sequence_data.plot_legend()
    
    # Most frequent sequences
    plot_most_frequent_sequences(sequence_data, top_n=5)
    
    # Mean time plot
    plot_mean_time(sequence_data)
    
    # State distribution
    plot_state_distribution(sequence_data)
    
    # Modal state
    plot_modal_state(sequence_data)
    
    # Transition matrix
    plot_transition_matrix(sequence_data)
    
    # If we reach here without errors, visualizations work
    assert True


def test_distance_matrix_computation():
    """Test distance matrix computation with different methods"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    
    # Test OM with TRATE substitution matrix
    om = get_distance_matrix(
        seqdata=sequence_data,
        method="OM",
        sm="TRATE",
        indel="auto"
    )
    
    assert om is not None
    assert isinstance(om, (np.ndarray, pd.DataFrame))
    assert om.shape[0] == om.shape[1]  # Should be square matrix
    assert om.shape[0] == sequence_data.num_sequences


def test_clustering_workflow():
    """Test the complete clustering workflow"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    
    # Compute distance matrix
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    
    # Create cluster object
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    assert cluster is not None
    
    # Test dendrogram plotting (without saving)
    import matplotlib
    matplotlib.use('Agg')
    cluster.plot_dendrogram(xlabel="Countries", ylabel="Distance")
    
    assert True


def test_cluster_quality_evaluation():
    """Test cluster quality evaluation metrics"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    
    # Create ClusterQuality object
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    
    # Get CQI table
    summary_table = cluster_quality.get_cqi_table()
    assert summary_table is not None
    assert isinstance(summary_table, pd.DataFrame)
    assert len(summary_table) > 0
    
    # Test plotting (without saving)
    import matplotlib
    matplotlib.use('Agg')
    cluster_quality.plot_cqi_scores(norm='zscore')
    
    assert True


def test_cluster_results_and_membership():
    """Test cluster results and membership extraction"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    
    # Create ClusterResults object
    cluster_results = ClusterResults(cluster)
    
    # Get cluster memberships
    membership_table = cluster_results.get_cluster_memberships(num_clusters=5)
    assert membership_table is not None
    assert isinstance(membership_table, pd.DataFrame)
    assert len(membership_table) == sequence_data.num_sequences
    assert 'Cluster' in membership_table.columns
    
    # Get cluster distribution
    distribution = cluster_results.get_cluster_distribution(num_clusters=5)
    assert distribution is not None
    assert isinstance(distribution, pd.DataFrame)
    assert len(distribution) == 5  # Should have 5 clusters
    
    # Test plotting (without saving)
    import matplotlib
    matplotlib.use('Agg')
    cluster_results.plot_cluster_distribution(num_clusters=5, title="Test Distribution")
    
    assert True


def test_grouped_visualizations():
    """Test visualizations with cluster grouping"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    cluster_results = ClusterResults(cluster)
    membership_table = cluster_results.get_cluster_memberships(num_clusters=5)
    
    cluster_labels = {
        1: 'Stable High',
        2: 'Steep Growth',
        3: 'Steady Growth',
        4: 'Rapid Growth',
        5: 'Persistent Low',
    }
    
    import matplotlib
    matplotlib.use('Agg')
    
    # Test index plot with grouping
    plot_sequence_index(
        seqdata=sequence_data,
        group_dataframe=membership_table,
        group_column_name="Cluster",
        group_labels=cluster_labels
    )
    
    # Test state distribution with grouping
    plot_state_distribution(
        seqdata=sequence_data,
        group_dataframe=membership_table,
        group_column_name="Cluster",
        group_labels=cluster_labels
    )
    
    assert True


def test_complete_workflow():
    """
    Test the complete workflow from data loading to final analysis
    This simulates what a real user would do following the quickstart tutorial
    """
    import matplotlib
    matplotlib.use('Agg')
    
    # Step 1: Load data
    df = load_dataset('country_co2_emissions_global_deciles')
    assert df is not None
    
    # Step 2: Create SequenceData
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    assert sequence_data is not None
    
    # Step 3: Visualizations
    plot_sequence_index(sequence_data)
    plot_state_distribution(sequence_data)
    
    # Step 4: Compute distance matrix
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    assert om is not None
    
    # Step 5: Cluster analysis
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    assert cluster is not None
    
    # Step 6: Evaluate cluster quality
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    summary_table = cluster_quality.get_cqi_table()
    assert len(summary_table) > 0
    
    # Step 7: Extract cluster memberships
    cluster_results = ClusterResults(cluster)
    membership_table = cluster_results.get_cluster_memberships(num_clusters=5)
    assert len(membership_table) == sequence_data.num_sequences
    
    # Step 8: Grouped visualizations
    cluster_labels = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5'}
    plot_sequence_index(seqdata=sequence_data, group_dataframe=membership_table, 
                       group_column_name="Cluster", group_labels=cluster_labels)
    
    # If we reach here, the complete workflow works!
    print("✓ Complete workflow test passed successfully!")
    assert True


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

