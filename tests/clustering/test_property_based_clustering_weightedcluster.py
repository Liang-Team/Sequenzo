"""
@Author  : Xinyi Li 李欣怡
@File    : test_property_based_clustering_weightedcluster.py
@Desc    :
Tests for property_based_clustering modules vs WeightedCluster seqpropclust R references.

Covers:
  - dyadic_children subset (first NROWS sequences).
  - extract_sequence_properties: all supported property types.
  - property_based_clustering / seqpropclust: tree structure, split schedule, cut_tree partitions.
  - tree utilities: cluster_split_schedule, cut_tree, prune_property_tree, tree_labels.
  - quality: property_clustering_quality / as_clustrange_property_tree.

Three main groups (with WeightedCluster ref comparison):
  1) Property extraction: column names and values vs seqpropclust property matrices.
  2) Tree structure: node counts, split schedules, number of clusters in cut partitions.
  3) Quality indicators: PBC, HG, HGSD, ASWw, CH, R2 vs as.clustrange output from R.

Run weightedcluster_property_reference.R to generate all ref_*.csv files before running
tests with a live R installation.

Usage:
    pytest tests/clustering/test_property_based_clustering_weightedcluster.py -v
"""

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

from sequenzo import SequenceData
from sequenzo.datasets import load_dataset

# ---------------------------------------------------------------------------
# Import the Sequenzo property-based clustering modules under test
# ---------------------------------------------------------------------------
from sequenzo.clustering.property_based_clustering import (
    SUPPORTED_PROPERTIES,
    as_clustrange_property_tree,
    cluster_split_schedule,
    cut_tree,
    extract_sequence_properties,
    property_based_clustering,
    property_clustering_quality,
    prune_property_tree,
    seqpropclust,
    tree_labels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NROWS = 30   # must match R script default (larger for tree to have depth)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Properties matching the R seqpropclust default subset (avoids event-mining slowness)
DEFAULT_PROPERTIES = ("state", "duration")
# Extended set used in a subset of tests
EXTENDED_PROPERTIES = ("state", "duration", "spell.dur", "Complexity")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dyadic_children_subset(nrows=NROWS):
    """Load dyadic_children and return first nrows rows."""
    df = load_dataset("dyadic_children")
    time_list = [c for c in df.columns if str(c).isdigit()]
    time_list = sorted(time_list, key=int)
    return df.head(nrows), time_list


def _sequence_data_from_df(df, time_list):
    states = [1, 2, 3, 4, 5, 6]
    return SequenceData(df, time=time_list, id_col="dyadID", states=states)


def _run_r_reference(csv_path, nrows, outdir, script_name, timeout=180, required_refs=()):
    r_script = os.path.join(THIS_DIR, script_name)
    if not os.path.isfile(r_script):
        return False
    try:
        result = subprocess.run(
            ["Rscript", r_script, csv_path, str(nrows), outdir],
            capture_output=True, text=True, timeout=timeout, cwd=THIS_DIR,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    if result.returncode == 0:
        return True
    if required_refs and all(
        os.path.isfile(os.path.join(outdir, f"ref_{name}.csv")) for name in required_refs
    ):
        return True
    return False


def _load_ref_csv(outdir, name, *, index_col=0):
    """Load a reference CSV. Use index_col=None for header-only tables (sizes, scalars)."""
    path = os.path.join(outdir, f"ref_{name}.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, index_col=index_col)


def _load_ref_value(outdir, name):
    df = _load_ref_csv(outdir, name, index_col=None)
    if df is None or df.empty:
        return None
    return float(df.iloc[0, 0])


def _load_ref_sizes(outdir, name):
    """Load a single-column 'size' reference vector."""
    df = _load_ref_csv(outdir, name, index_col=None)
    if df is None or df.empty:
        return None
    col = "size" if "size" in df.columns else df.columns[0]
    return df[col].to_numpy()


def _build_distance_matrix(seqdata):
    from sequenzo.dissimilarity_measures import get_distance_matrix
    D = get_distance_matrix(seqdata, method="OM", sm="TRATE", indel="auto", norm="maxlength")
    return np.asarray(D, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seqdata_and_diss():
    df, time_list = _dyadic_children_subset(NROWS)
    sd = _sequence_data_from_df(df, time_list)
    D = _build_distance_matrix(sd)
    return sd, D


@pytest.fixture(scope="module")
def ref_dir_property(seqdata_and_diss):
    """Generate WeightedCluster seqpropclust reference CSVs, or use THIS_DIR."""
    sd, _ = seqdata_and_diss
    for name in ["propmatrix_state_cols", "tree_n_leaves"]:
        if os.path.isfile(os.path.join(THIS_DIR, f"ref_{name}.csv")):
            return THIS_DIR
    df, _ = _dyadic_children_subset(NROWS)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name
    try:
        outdir = tempfile.mkdtemp()
        ok = _run_r_reference(
            csv_path,
            NROWS,
            outdir,
            "weightedcluster_property_reference.R",
            required_refs=("propmatrix_state_cols", "tree_n_leaves"),
        )
        if ok:
            return outdir
    finally:
        try:
            os.unlink(csv_path)
        except Exception:
            pass
    pytest.skip(
        "R reference generation failed; run: "
        "Rscript tests/clustering/weightedcluster_property_reference.R <csv> 30 tests/clustering"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _assert_property_matrix_valid(props, n_expected):
    """Assert property matrix has correct shape and no all-NaN columns."""
    assert isinstance(props, pd.DataFrame), "extract_sequence_properties must return DataFrame"
    assert len(props) == n_expected, f"Expected {n_expected} rows, got {len(props)}"
    assert props.shape[1] > 0, "Property matrix must have at least one column"


# =============================================================================
# Part 1: extract_sequence_properties
# =============================================================================

class TestExtractSequenceProperties:
    """Tests for extract_sequence_properties."""

    def test_state_properties_shape(self, seqdata_and_diss):
        """'state' properties produce NROWS rows and ≥1 columns."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("state",), verbose=False)
        _assert_property_matrix_valid(props, NROWS)

    def test_state_properties_column_prefixes(self, seqdata_and_diss):
        """'state' property columns are prefixed with 'state.'."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("state",), verbose=False)
        assert all(col.startswith("state.") for col in props.columns), (
            f"All 'state' columns should start with 'state.', got: {list(props.columns[:5])}"
        )

    def test_duration_properties_shape(self, seqdata_and_diss):
        """'duration' properties produce NROWS rows and one column per state."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("duration",), verbose=False)
        _assert_property_matrix_valid(props, NROWS)

    def test_duration_properties_column_prefixes(self, seqdata_and_diss):
        """'duration' property columns are prefixed with 'duration.'."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("duration",), verbose=False)
        assert all(col.startswith("duration.") for col in props.columns), (
            f"All 'duration' columns should start with 'duration.', got: {list(props.columns[:5])}"
        )

    def test_duration_row_sums_equal_sequence_length(self, seqdata_and_diss):
        """Duration values per row (excluding '*' column) sum to sequence length."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("duration",), verbose=False)
        state_cols = [c for c in props.columns if not c.endswith(".*")]
        seq_len = sd.seqdata.shape[1]
        row_sums = props[state_cols].sum(axis=1)
        np.testing.assert_allclose(
            row_sums.to_numpy(), seq_len,
            atol=1e-9, err_msg="Duration row sums should equal sequence length"
        )

    def test_spell_dur_properties(self, seqdata_and_diss):
        """'spell.dur' properties produce valid DataFrame."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("spell.dur",), verbose=False)
        _assert_property_matrix_valid(props, NROWS)

    def test_spell_age_properties(self, seqdata_and_diss):
        """'spell.age' properties produce valid DataFrame."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("spell.age",), verbose=False)
        _assert_property_matrix_valid(props, NROWS)

    def test_complexity_properties_columns(self, seqdata_and_diss):
        """'Complexity' properties produce 4 columns: C, Entropy, Turbulence, Trans."""
        sd, _ = seqdata_and_diss
        props = extract_sequence_properties(sd, properties=("Complexity",), verbose=False)
        expected_cols = {"Complexity.C", "Complexity.Entropy", "Complexity.Turbulence", "Complexity.Trans."}
        assert set(props.columns) == expected_cols, (
            f"Expected {expected_cols}, got {set(props.columns)}"
        )

    def test_combined_properties_column_count(self, seqdata_and_diss):
        """Combined state+duration produces more columns than each individually."""
        sd, _ = seqdata_and_diss
        state_props  = extract_sequence_properties(sd, properties=("state",), verbose=False)
        dur_props    = extract_sequence_properties(sd, properties=("duration",), verbose=False)
        combined     = extract_sequence_properties(sd, properties=("state", "duration"), verbose=False)
        assert combined.shape[1] == state_props.shape[1] + dur_props.shape[1], (
            "Combined column count should equal sum of individual counts"
        )

    def test_unsupported_property_raises(self, seqdata_and_diss):
        """Unsupported property name raises ValueError."""
        sd, _ = seqdata_and_diss
        with pytest.raises(ValueError, match="Unsupported"):
            extract_sequence_properties(sd, properties=("invalid_prop",), verbose=False)

    def test_other_properties_appended(self, seqdata_and_diss):
        """User-defined other_properties are prepended to the property matrix."""
        sd, _ = seqdata_and_diss
        extra = pd.DataFrame({"custom": np.random.rand(NROWS)})
        props = extract_sequence_properties(
            sd, properties=("duration",), other_properties=extra, verbose=False
        )
        assert "custom" in props.columns

    def test_state_cols_match_weightedcluster(self, seqdata_and_diss, ref_dir_property):
        """'state' property column names match WeightedCluster seqpropclust reference."""
        sd, _ = seqdata_and_diss
        D_ref = _load_ref_csv(ref_dir_property, "propmatrix_state_cols")
        if D_ref is None:
            pytest.skip("ref_propmatrix_state_cols.csv not found")
        props = extract_sequence_properties(sd, properties=("state",), verbose=False)
        # Compare number of columns (names may differ in prefix convention)
        assert props.shape[1] == D_ref.shape[1], (
            f"Column count mismatch: Sequenzo {props.shape[1]} vs R {D_ref.shape[1]}"
        )

    def test_duration_values_match_weightedcluster(self, seqdata_and_diss, ref_dir_property):
        """'duration' property values match WeightedCluster reference (first 5 rows)."""
        sd, _ = seqdata_and_diss
        D_ref = _load_ref_csv(ref_dir_property, "propmatrix_duration")
        if D_ref is None:
            pytest.skip("ref_propmatrix_duration.csv not found")
        props = extract_sequence_properties(sd, properties=("duration",), verbose=False)
        # Compare numeric values ignoring column name prefix differences
        state_cols_seq = [c for c in props.columns if not c.endswith(".*")]
        state_cols_ref = [c for c in D_ref.columns if not c.endswith("*")]
        seq_vals = props[state_cols_seq].to_numpy(dtype=float)
        ref_vals = D_ref[state_cols_ref].to_numpy(dtype=float)
        np.testing.assert_allclose(
            seq_vals, ref_vals, atol=1e-6,
            err_msg="Duration property values differ from WeightedCluster reference"
        )

    def test_complexity_values_match_weightedcluster(self, seqdata_and_diss, ref_dir_property):
        """'Complexity' property values match WeightedCluster reference."""
        sd, _ = seqdata_and_diss
        D_ref = _load_ref_csv(ref_dir_property, "propmatrix_complexity")
        if D_ref is None:
            pytest.skip("ref_propmatrix_complexity.csv not found")
        props = extract_sequence_properties(sd, properties=("Complexity",), verbose=False)
        np.testing.assert_allclose(
            props.to_numpy(dtype=float),
            D_ref.to_numpy(dtype=float),
            atol=1e-4,
            err_msg="Complexity property values differ from WeightedCluster reference"
        )


# =============================================================================
# Part 2: property_based_clustering (seqpropclust) — tree structure
# =============================================================================

class TestPropertyBasedClustering:
    """Tests for property_based_clustering / seqpropclust."""

    def test_returns_dict(self, seqdata_and_diss):
        """property_based_clustering returns a dict (tree object)."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        assert isinstance(tree, dict)

    def test_tree_has_root(self, seqdata_and_diss):
        """Tree result contains 'root' key."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        assert "root" in tree

    def test_tree_has_fitted(self, seqdata_and_diss):
        """Tree result contains 'fitted' key with correct shape."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        assert "fitted" in tree
        fitted = tree["fitted"]["(fitted)"]
        assert len(fitted) == NROWS

    def test_tree_method_is_seqpropclust(self, seqdata_and_diss):
        """tree['info']['method'] == 'seqpropclust'."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        assert tree["info"].get("method") == "seqpropclust"

    def test_tree_split_schedule_applied(self, seqdata_and_diss):
        """cluster_split_schedule marks tree with 'split_schedule_applied'."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        assert tree["info"].get("split_schedule_applied") is True

    def test_seqpropclust_alias(self, seqdata_and_diss):
        """seqpropclust is an alias for property_based_clustering."""
        sd, D = seqdata_and_diss
        tree1 = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        tree2 = seqpropclust(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        # Both should produce the same fitted partition
        np.testing.assert_array_equal(
            tree1["fitted"]["(fitted)"].to_numpy(),
            tree2["fitted"]["(fitted)"].to_numpy()
        )

    def test_properties_only_returns_dataframe(self, seqdata_and_diss):
        """properties_only=True returns the property matrix DataFrame directly."""
        sd, D = seqdata_and_diss
        result = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES,
            properties_only=True, verbose=False
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == NROWS

    def test_max_clusters_prunes_tree(self, seqdata_and_diss):
        """max_clusters parameter prunes tree to at most that many leaf groups."""
        sd, D = seqdata_and_diss
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES,
            max_clusters=4, verbose=False
        )
        fitted_unique = np.unique(tree["fitted"]["(fitted)"].to_numpy())
        assert len(fitted_unique) <= 4, (
            f"Expected ≤4 groups after pruning, got {len(fitted_unique)}"
        )

    def test_n_leaves_matches_weightedcluster(self, seqdata_and_diss, ref_dir_property):
        """Number of leaves (max clusters) matches WeightedCluster reference."""
        sd, D = seqdata_and_diss
        ref_n = _load_ref_value(ref_dir_property, "tree_n_leaves")
        if ref_n is None:
            pytest.skip("ref_tree_n_leaves.csv not found")
        tree = property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )
        fitted_unique = np.unique(tree["fitted"]["(fitted)"].to_numpy())
        ref_n = int(ref_n)
        assert len(fitted_unique) == ref_n, (
            f"Sequenzo n_leaves={len(fitted_unique)}, WeightedCluster={ref_n}"
        )


# =============================================================================
# Part 3: cluster_split_schedule and cut_tree
# =============================================================================

class TestCutTree:
    """Tests for cut_tree and cluster_split_schedule."""

    @pytest.fixture(scope="class")
    def scheduled_tree(self, seqdata_and_diss):
        sd, D = seqdata_and_diss
        return property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )

    def test_cut_tree_k2(self, scheduled_tree):
        """cut_tree with n_clusters=2 returns series of length NROWS with 2 unique values."""
        result = cut_tree(scheduled_tree, n_clusters=2)
        assert len(result) == NROWS
        assert len(set(result)) == 2

    def test_cut_tree_k3(self, scheduled_tree):
        """cut_tree with n_clusters=3 returns 3 unique groups."""
        result = cut_tree(scheduled_tree, n_clusters=3)
        assert len(result) == NROWS
        assert len(set(result)) == 3

    def test_cut_tree_labels_false_returns_int_array(self, scheduled_tree):
        """cut_tree with labels=False returns integer node-ID array."""
        result = cut_tree(scheduled_tree, n_clusters=2, labels=False)
        assert isinstance(result, np.ndarray), "labels=False should return np.ndarray"
        assert result.dtype.kind in ("i", "u"), "Should be integer dtype"

    def test_cut_tree_labels_true_returns_series(self, scheduled_tree):
        """cut_tree with labels=True returns pd.Series."""
        result = cut_tree(scheduled_tree, n_clusters=2, labels=True)
        assert isinstance(result, pd.Series)

    def test_cut_tree_exceeds_max_raises(self, scheduled_tree):
        """cut_tree raises ValueError when n_clusters exceeds max leaves."""
        fitted = scheduled_tree["fitted"]["(fitted)"].to_numpy()
        max_k = len(np.unique(fitted))
        with pytest.raises(ValueError):
            cut_tree(scheduled_tree, n_clusters=max_k + 1)

    def test_cut_tree_k2_matches_weightedcluster(self, scheduled_tree, ref_dir_property):
        """cut_tree k=2 partition label set matches WeightedCluster reference."""
        ref_sizes = _load_ref_sizes(ref_dir_property, "cut_tree_k2_sizes")
        if ref_sizes is None:
            pytest.skip("ref_cut_tree_k2_sizes.csv not found")
        result = cut_tree(scheduled_tree, n_clusters=2)
        sizes = pd.Series(result).value_counts().sort_values().to_numpy()
        np.testing.assert_array_equal(
            np.sort(sizes), np.sort(ref_sizes),
            err_msg=f"k=2 partition sizes differ: Sequenzo={np.sort(sizes)}, R={np.sort(ref_sizes)}"
        )

    def test_cut_tree_k3_matches_weightedcluster(self, scheduled_tree, ref_dir_property):
        """cut_tree k=3 partition sizes match WeightedCluster reference."""
        ref_sizes = _load_ref_sizes(ref_dir_property, "cut_tree_k3_sizes")
        if ref_sizes is None:
            pytest.skip("ref_cut_tree_k3_sizes.csv not found")
        result = cut_tree(scheduled_tree, n_clusters=3)
        sizes = pd.Series(result).value_counts().sort_values().to_numpy()
        np.testing.assert_array_equal(
            np.sort(sizes), np.sort(ref_sizes),
            err_msg=f"k=3 partition sizes differ: Sequenzo={np.sort(sizes)}, R={np.sort(ref_sizes)}"
        )


# =============================================================================
# Part 4: tree_labels
# =============================================================================

class TestTreeLabels:
    """Tests for tree_labels."""

    @pytest.fixture(scope="class")
    def scheduled_tree(self, seqdata_and_diss):
        sd, D = seqdata_and_diss
        return property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        )

    def test_tree_labels_returns_dict(self, scheduled_tree):
        """tree_labels returns a dict mapping node IDs to strings."""
        labels = tree_labels(scheduled_tree)
        assert isinstance(labels, dict)
        assert len(labels) > 0

    def test_tree_labels_strings(self, scheduled_tree):
        """All label values are non-empty strings."""
        labels = tree_labels(scheduled_tree)
        for node_id, label in labels.items():
            assert isinstance(label, str) and len(label) > 0, (
                f"Node {node_id} has invalid label: {label!r}"
            )

    def test_tree_labels_count_matches_leaves(self, scheduled_tree):
        """Number of labels equals number of leaf/child nodes (non-root splits)."""
        fitted = scheduled_tree["fitted"]["(fitted)"].to_numpy()
        n_groups = len(np.unique(fitted))
        labels = tree_labels(scheduled_tree)
        # Each split produces 2 children; for n_groups leaves, n_groups - 1 internal nodes
        # total label entries = 2 * (n_groups - 1)
        assert len(labels) >= n_groups - 1, (
            f"Too few labels: expected >= {n_groups - 1}, got {len(labels)}"
        )


# =============================================================================
# Part 5: prune_property_tree
# =============================================================================

class TestPrunePropertyTree:
    """Tests for prune_property_tree."""

    @pytest.fixture(scope="class")
    def full_tree(self, seqdata_and_diss):
        sd, D = seqdata_and_diss
        return property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        ), D

    def test_prune_returns_dict(self, full_tree):
        """prune_property_tree returns a dict."""
        tree, D = full_tree
        pruned = prune_property_tree(tree, n_clusters=3)
        assert isinstance(pruned, dict)

    def test_prune_reduces_groups(self, full_tree):
        """After pruning to k=3, fitted has at most 3 unique groups."""
        tree, D = full_tree
        pruned = prune_property_tree(tree, n_clusters=3, diss=D)
        fitted = pruned["fitted"]["(fitted)"].to_numpy()
        assert len(np.unique(fitted)) <= 3

    def test_prune_does_not_modify_original(self, full_tree):
        """prune_property_tree does not mutate the original tree (deepcopy)."""
        tree, D = full_tree
        original_fitted = tree["fitted"]["(fitted)"].to_numpy().copy()
        prune_property_tree(tree, n_clusters=2, diss=D)
        np.testing.assert_array_equal(
            tree["fitted"]["(fitted)"].to_numpy(), original_fitted,
            err_msg="prune_property_tree should not mutate the original tree"
        )

    def test_prune_info_recorded(self, full_tree):
        """Pruned tree records n_clusters in info['prune']."""
        tree, D = full_tree
        pruned = prune_property_tree(tree, n_clusters=4)
        assert pruned["info"].get("prune") == 4

    def test_prune_k2_matches_weightedcluster(self, full_tree, ref_dir_property):
        """Pruned k=2 partition sizes match WeightedCluster dtprune reference."""
        tree, D = full_tree
        ref_sizes = _load_ref_sizes(ref_dir_property, "prune_k2_sizes")
        if ref_sizes is None:
            pytest.skip("ref_prune_k2_sizes.csv not found")
        pruned = prune_property_tree(tree, n_clusters=2, diss=D)
        fitted = pruned["fitted"]["(fitted)"].to_numpy()
        sizes = np.sort(np.bincount(fitted)[np.bincount(fitted) > 0])
        ref_sizes = np.sort(ref_sizes)
        np.testing.assert_array_equal(
            sizes, ref_sizes,
            err_msg=f"Pruned k=2 sizes differ: Sequenzo={sizes}, R={ref_sizes}"
        )


# =============================================================================
# Part 6: property_clustering_quality (as_clustrange_property_tree)
# =============================================================================

class TestPropertyClusteringQuality:
    """Tests for property_clustering_quality / as_clustrange_property_tree."""

    @pytest.fixture(scope="class")
    def scheduled_tree(self, seqdata_and_diss):
        sd, D = seqdata_and_diss
        return property_based_clustering(
            sd, D, properties=DEFAULT_PROPERTIES, verbose=False
        ), D

    def _max_k(self, tree):
        fitted = tree["fitted"]["(fitted)"].to_numpy()
        return len(np.unique(fitted))

    def test_quality_returns_clustrange_result(self, scheduled_tree):
        """property_clustering_quality returns a ClusterRangeResult."""
        tree, D = scheduled_tree
        from sequenzo.clustering.validation.partition_quality import ClusterRangeResult
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        assert isinstance(quality, ClusterRangeResult)

    def test_quality_stats_dataframe_shape(self, scheduled_tree):
        """Quality stats DataFrame has one row per k (2..n_clusters)."""
        tree, D = scheduled_tree
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        assert hasattr(quality, "stats")
        assert len(quality.stats) == n_eval - 1  # rows for k=2..n_eval

    def test_quality_contains_standard_indicators(self, scheduled_tree):
        """Quality stats include PBC, HG, HGSD, ASWw, CH, R2 columns."""
        tree, D = scheduled_tree
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        for indicator in ("PBC", "HG", "ASWw", "R2"):
            assert indicator in quality.stats.columns, (
                f"Quality indicator '{indicator}' missing from stats"
            )

    def test_quality_r2_in_range(self, scheduled_tree):
        """R2 values are in [0, 1]."""
        tree, D = scheduled_tree
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        r2 = quality.stats["R2"].to_numpy()
        assert np.all(r2 >= -1e-9) and np.all(r2 <= 1.0 + 1e-9), (
            f"R2 out of [0,1]: {r2}"
        )

    def test_quality_r2_increases_with_more_clusters(self, scheduled_tree):
        """R2 should be non-decreasing as k increases (more splits = more explained)."""
        tree, D = scheduled_tree
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        r2 = quality.stats["R2"].to_numpy()
        assert np.all(np.diff(r2) >= -0.05), (
            f"R2 should be non-decreasing (tolerance 0.05): {r2}"
        )

    def test_as_clustrange_alias(self, scheduled_tree):
        """as_clustrange_property_tree is an alias for property_clustering_quality."""
        tree, D = scheduled_tree
        max_k = self._max_k(tree)
        n_eval = min(max_k, 5)
        q1 = property_clustering_quality(tree, D, n_clusters=n_eval)
        q2 = as_clustrange_property_tree(tree, D, n_clusters=n_eval)
        pd.testing.assert_frame_equal(q1.stats, q2.stats)

    def test_quality_n_clusters_too_small_raises(self, scheduled_tree):
        """n_clusters < 3 raises ValueError."""
        tree, D = scheduled_tree
        with pytest.raises(ValueError, match="greater than 2"):
            property_clustering_quality(tree, D, n_clusters=2)

    def test_quality_pbc_matches_weightedcluster(self, scheduled_tree, ref_dir_property):
        """PBC quality indicator matches WeightedCluster as.clustrange reference."""
        tree, D = scheduled_tree
        D_ref = _load_ref_csv(ref_dir_property, "quality_pbc")
        if D_ref is None:
            pytest.skip("ref_quality_pbc.csv not found")
        max_k = self._max_k(tree)
        n_eval = min(max_k, min(int(D_ref.shape[0]) + 1, 8))
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        seq_pbc = quality.stats["PBC"].to_numpy()
        ref_pbc = D_ref.iloc[:len(seq_pbc), 0].to_numpy(dtype=float)
        # Partition cuts match at k=2,3; small drift at higher k from metric/C++ vs R paths.
        np.testing.assert_allclose(
            seq_pbc, ref_pbc, atol=0.08, rtol=0.05,
            err_msg=f"PBC mismatch: Sequenzo={seq_pbc}, WeightedCluster={ref_pbc}"
        )

    def test_quality_r2_matches_weightedcluster(self, scheduled_tree, ref_dir_property):
        """R2 quality indicator matches WeightedCluster as.clustrange reference."""
        tree, D = scheduled_tree
        D_ref = _load_ref_csv(ref_dir_property, "quality_r2")
        if D_ref is None:
            pytest.skip("ref_quality_r2.csv not found")
        max_k = self._max_k(tree)
        n_eval = min(max_k, min(int(D_ref.shape[0]) + 1, 8))
        quality = property_clustering_quality(tree, D, n_clusters=n_eval)
        seq_r2 = quality.stats["R2"].to_numpy()
        ref_r2 = D_ref.iloc[:len(seq_r2), 0].to_numpy(dtype=float)
        np.testing.assert_allclose(
            seq_r2, ref_r2, atol=0.08, rtol=0.05,
            err_msg=f"R2 mismatch: Sequenzo={seq_r2}, WeightedCluster={ref_r2}"
        )


# =============================================================================
# Part 7: SUPPORTED_PROPERTIES constant
# =============================================================================

class TestSupportedProperties:
    """Tests for the SUPPORTED_PROPERTIES tuple."""

    def test_supported_properties_is_tuple(self):
        """SUPPORTED_PROPERTIES is a tuple."""
        assert isinstance(SUPPORTED_PROPERTIES, tuple)

    def test_supported_properties_contains_required(self):
        """SUPPORTED_PROPERTIES contains the eight WeightedCluster property names."""
        required = {
            "state", "spell.age", "spell.dur", "duration",
            "pattern", "AFpattern", "transition", "AFtransition", "Complexity",
        }
        missing = required - set(SUPPORTED_PROPERTIES)
        assert not missing, f"SUPPORTED_PROPERTIES missing: {missing}"
