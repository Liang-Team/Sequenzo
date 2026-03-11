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
    assert sequence_data.n_sequences > 0
    assert sequence_data.n_steps > 0
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
    assert om.shape[0] == sequence_data.n_sequences


def test_clustering_workflow():
    """Test the complete clustering workflow"""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']
    
    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    
    # Compute distance matrix
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    
    # Create cluster object (using ward_d2 for CI compatibility)
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')
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
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')
    
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
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')
    
    # Create ClusterResults object
    cluster_results = ClusterResults(cluster)
    
    # Get cluster memberships
    membership_table = cluster_results.get_cluster_memberships(num_clusters=5)
    assert membership_table is not None
    assert isinstance(membership_table, pd.DataFrame)
    assert len(membership_table) == sequence_data.n_sequences
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
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')
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
    
    # Step 5: Cluster analysis (using ward_d2 for CI compatibility)
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')
    assert cluster is not None
    
    # Step 6: Evaluate cluster quality
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    summary_table = cluster_quality.get_cqi_table()
    assert len(summary_table) > 0
    
    # Step 7: Extract cluster memberships
    cluster_results = ClusterResults(cluster)
    membership_table = cluster_results.get_cluster_memberships(num_clusters=5)
    assert len(membership_table) == sequence_data.n_sequences
    
    # Step 8: Grouped visualizations
    cluster_labels = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4', 5: 'Cluster 5'}
    plot_sequence_index(seqdata=sequence_data, group_dataframe=membership_table, 
                       group_column_name="Cluster", group_labels=cluster_labels)
    
    # If we reach here, the complete workflow works!
    print("OK - Complete workflow test passed successfully!")
    assert True


def test_cluster_quality_direct_matrix_path():
    """Test ClusterQuality initialised with a raw distance matrix (path B)."""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']

    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')

    cq_from_cluster = ClusterQuality(cluster, max_clusters=10)
    cq_from_matrix = ClusterQuality(om, max_clusters=10, clustering_method='ward_d2')

    for metric in cq_from_cluster.metric_order:
        a = np.array(cq_from_cluster.scores[metric], dtype=np.float64)
        b = np.array(cq_from_matrix.scores[metric], dtype=np.float64)
        assert np.allclose(a, b, atol=1e-10, equal_nan=True), \
            f"Direct-matrix path diverges on {metric}"

    t1 = cq_from_cluster.get_cqi_table()
    t2 = cq_from_matrix.get_cqi_table()
    assert np.allclose(t1["Opt. Clusters"].values, t2["Opt. Clusters"].values, equal_nan=True)
    assert np.allclose(t1["Raw Value"].values, t2["Raw Value"].values, atol=1e-10, equal_nan=True)

    r1 = cq_from_cluster.get_cluster_range_table()
    r2 = cq_from_matrix.get_cluster_range_table()
    assert np.allclose(r1.values, r2.values, atol=1e-10, equal_nan=True)


def test_cluster_quality_ward_d():
    """Test ClusterQuality with ward_d method."""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']

    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")

    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    cq = ClusterQuality(cluster, max_clusters=10)

    table = cq.get_cqi_table()
    assert table is not None
    assert len(table) == 10
    assert all(table["Opt. Clusters"].notna())

    range_table = cq.get_cluster_range_table()
    assert range_table.shape == (9, 10)  # k=2..10 => 9 rows, 10 metrics


def test_weighted_cluster_results():
    """Test ClusterResults with weighted data."""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']

    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")

    n = om.shape[0] if isinstance(om, np.ndarray) else om.values.shape[0]
    rng = np.random.RandomState(42)
    weights = rng.uniform(0.5, 2.0, size=n)

    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2', weights=weights)
    cr = ClusterResults(cluster)

    dist = cr.get_cluster_distribution(num_clusters=5, weighted=True)
    assert 'Weight_Sum' in dist.columns
    assert 'Weight_Percentage' in dist.columns
    assert np.isclose(dist['Weight_Percentage'].sum(), 100.0, atol=0.1)
    assert np.isclose(dist['Percentage'].sum(), 100.0, atol=0.1)

    dist_unweighted = cr.get_cluster_distribution(num_clusters=5, weighted=False)
    assert 'Weight_Sum' not in dist_unweighted.columns


def test_cluster_quality_compute_scores_noop():
    """Test that compute_cluster_quality_scores() is a no-op (scores computed in __init__)."""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']

    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')

    cq = ClusterQuality(cluster, max_clusters=10)
    scores_before = {m: list(cq.scores[m]) for m in cq.metric_order}
    cq.compute_cluster_quality_scores()
    for m in cq.metric_order:
        assert cq.scores[m] == scores_before[m], f"compute_cluster_quality_scores() changed {m}"


def test_cluster_results_caching():
    """Test that repeated calls to get_cluster_memberships use cache."""
    df = load_dataset('country_co2_emissions_global_deciles')
    time_list = list(df.columns)[1:]
    states = ['D1 (Very Low)', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10 (Very High)']

    sequence_data = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=sequence_data, method="OM", sm="TRATE", indel="auto")
    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d2')

    cr = ClusterResults(cluster)
    m1 = cr.get_cluster_memberships(5)
    m2 = cr.get_cluster_memberships(5)
    assert np.array_equal(m1["Cluster"].values, m2["Cluster"].values)
    assert 5 in cr._results_cache


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])

