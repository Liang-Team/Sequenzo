# Sequenzo Test Suite

This directory contains the comprehensive test suite for the Sequenzo package. The tests are designed to ensure code quality, functionality correctness, and compatibility with TraMineR reference implementations.

## Table of Contents

- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Environment Setup](#test-environment-setup)
- [Test Categories](#test-categories)
- [CI/CD Integration](#cicd-integration)
- [Development Guidelines](#development-guidelines)
- [Troubleshooting](#troubleshooting)

## Test Structure

The test suite is organized into several categories:

```
tests/
├── __init__.py
├── README.md                          # This file
├── test_basic.py                       # Basic package tests
├── test_pam_and_kmedoids.py           # Clustering algorithm tests
├── test_quickstart_integration.py     # End-to-end integration tests
├── dissimilarity_measures/            # Distance measure tests
│   ├── test_dissimilarity_measures_traminer.py
│   └── ref_*.csv                       # TraMineR reference outputs
├── sequence_characteristics/          # Sequence characteristic tests
│   └── test_sequence_characteristics_lsog.py
├── event_sequence_analysis/            # Event sequence (TraMineR parity)
│   ├── README.md                       # Why 3 tests skip, how to generate refs
│   ├── test_event_sequence_lsog.py
│   └── traminer_reference_event_sequence.R
└── openmp/                            # OpenMP-related tests and solutions
    ├── test_apple_silicon_solution.py
    └── test_solution_simple.py
```

## Running Tests

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install pytest
pip install sequenzo  # Or use: pip install -e . for development mode
```

### Run All Tests

To run the entire test suite:

```bash
pytest tests/ -v
```

The `-v` flag enables verbose output, showing each test as it runs.

### Run Specific Test Files

#### Basic Tests
Tests for package imports, version checks, and basic functionality:

```bash
pytest tests/test_basic.py -v
```

#### Integration Tests (Recommended)
End-to-end tests that simulate real user workflows:

```bash
pytest tests/test_quickstart_integration.py -v -s
```

The `-s` flag allows print statements to be displayed, which is useful for debugging.

#### Clustering Algorithm Tests
Tests for PAM and K-medoids clustering algorithms:

```bash
pytest tests/test_pam_and_kmedoids.py -v
```

#### Distance Measure Tests
Tests for dissimilarity measures against TraMineR reference outputs:

```bash
pytest tests/dissimilarity_measures/test_dissimilarity_measures_traminer.py -v
```

#### Sequence Characteristics Tests
Tests for sequence characteristic calculations:

```bash
pytest tests/sequence_characteristics/test_sequence_characteristics_lsog.py -v
```

#### Event Sequence Analysis Tests
Tests for event sequence API (create_event_sequences, find_frequent_subsequences, etc.) using LSOG. **Three tests compare to TraMineR and are skipped unless reference files exist.** To generate refs and run all tests (including TraMineR parity), see `tests/event_sequence_analysis/README.md`.

```bash
pytest tests/event_sequence_analysis/test_event_sequence_lsog.py -v
```

### Run Individual Test Functions

You can run a specific test function by specifying its path:

```bash
# Run complete workflow test
pytest tests/test_quickstart_integration.py::test_complete_workflow -v -s

# Run visualization tests
pytest tests/test_quickstart_integration.py::test_visualizations_no_save -v -s

# Run a specific sequence characteristic test
pytest tests/sequence_characteristics/test_sequence_characteristics_lsog.py::test_get_spell_durations -v
```

### Run Tests with Coverage

To generate a coverage report:

```bash
pip install pytest-cov
pytest tests/ --cov=sequenzo --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

## Test Environment Setup

### Development Mode Installation

For development, install the package in editable mode:

```bash
pip install -e .
```

This allows you to modify the source code and see changes immediately without reinstalling.

### Virtual Environment (Recommended)

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv sequenzo_test

# Activate virtual environment
# On macOS/Linux:
source sequenzo_test/bin/activate
# On Windows:
sequenzo_test\Scripts\activate

# Install dependencies
pip install -e .
pip install pytest pytest-cov
```

## Test Categories

### 1. Basic Tests (`test_basic.py`)

**Purpose**: Verify fundamental package functionality

**Tests**:
- Package import verification
- Version number checks
- Basic object instantiation
- C++ extension loading

**When to run**: After any code changes to ensure basic functionality still works.

### 2. Integration Tests (`test_quickstart_integration.py`)

**Purpose**: Simulate real-world user workflows end-to-end

**Tests**:
- Dataset loading from various sources
- `SequenceData` object creation and validation
- Visualization functions (sequence plots, state distribution plots, etc.)
- Distance matrix calculation with various methods
- Clustering analysis (hierarchical, PAM, K-medoids)
- Clustering quality assessment metrics
- Complete user workflow from data loading to results

**When to run**: Before committing major changes or before releases.

**Why it matters**: These tests catch integration issues that unit tests might miss, ensuring the package works correctly in real usage scenarios.

### 3. Clustering Tests (`test_pam_and_kmedoids.py`)

**Purpose**: Validate clustering algorithm implementations

**Tests**:
- PAM (Partitioning Around Medoids) algorithm correctness
- K-medoids clustering functionality
- Cluster assignment accuracy
- Algorithm parameter handling

### 4. Distance Measure Tests (`dissimilarity_measures/`)

**Purpose**: Ensure distance calculations match TraMineR reference implementations

**Tests**:
- Multiple distance methods (OM, LCS, NMS, TWED, etc.)
- Comparison against TraMineR reference outputs (`ref_*.csv`)
- Parameter variations (indel costs, normalization, etc.)
- Edge cases and boundary conditions

**Reference Files**: The `ref_*.csv` files contain expected outputs from TraMineR R package, ensuring our Python implementation produces identical results.

### 5. Sequence Characteristics Tests (`sequence_characteristics/`)

**Purpose**: Validate sequence characteristic calculations

**Tests**:
- Basic indicators (sequence length, spell durations, visited states)
- Diversity indicators (entropy difference)
- Complexity indicators (volatility)
- Binary indicators (positive/negative indicators, integration index)
- Ranked indicators (badness, degradation, precarity, insecurity indices)
- Cross-sectional indicators (mean time in states, modal state sequence)

**Dataset**: Uses the `dyadic_children` (lsog) dataset for comprehensive testing.

### 6. Event Sequence Analysis Tests (`event_sequence_analysis/`)

**Purpose**: Validate event sequence analysis and ensure results match TraMineR.

**Tests**:
- Create event sequences from state data, find frequent subsequences, count occurrences, compare groups
- **TraMineR parity** (when ref files exist): meta (n_sequences, n_events), fsub Support/Count, applysub presence matrix
- Visualization: plot_event_sequences (index/parallel), plot_subsequence_frequencies

**Why 3 tests are skipped by default**: The three TraMineR comparison tests require reference files generated by R. Generate them from repo root with:
`Rscript tests/event_sequence_analysis/traminer_reference_event_sequence.R sequenzo/datasets/dyadic_children.csv 20 tests/event_sequence_analysis`
See `tests/event_sequence_analysis/README.md` for details.

### 7. OpenMP Tests (`openmp/`)

**Purpose**: Verify OpenMP parallelization works correctly

**Tests**:
- Apple Silicon compatibility
- Multi-threading correctness
- Performance improvements

**Documentation**: See `openmp/for_users_APPLE_SILICON_GUIDE.md` and `openmp/for_developers_OPENMP_SOLUTION_SUMMARY.md` for detailed information.

## CI/CD Integration

### GitHub Actions Workflow

The test suite is automatically run in GitHub Actions CI/CD pipeline:

1. **Build Phase**: Creates wheel packages for multiple platforms
2. **Test Phase**: Runs automated tests:
   - Basic import tests
   - C++ extension loading verification
   - Full integration test suite (`test_quickstart_integration.py`)
   - Distance measure validation against TraMineR references

### Why This Matters

Automated testing ensures that:
- Built wheel packages work correctly in real scenarios
- Code changes don't break existing functionality
- Cross-platform compatibility is maintained
- Issues are caught before users encounter them

## Development Guidelines

### Before Committing Code

Always run tests before committing:

```bash
# Quick sanity check
pytest tests/test_basic.py -v

# Full test suite (recommended)
pytest tests/ -v
```

### Writing New Tests

When adding new features, follow these guidelines:

1. **Add unit tests** for new functions/classes
2. **Add integration tests** if the feature affects user workflows
3. **Update reference files** if distance measures change
4. **Document test purpose** in docstrings

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test functions: `test_<functionality_description>`
- Test classes: `Test<ClassName>`

### Example Test Structure

```python
def test_feature_name(seqdata):
    """
    Test description explaining what this test validates.
    
    Args:
        seqdata: Fixture providing SequenceData object
    """
    # Arrange
    expected_result = ...
    
    # Act
    result = function_to_test(seqdata)
    
    # Assert
    assert result.equals(expected_result)
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'sequenzo'`

**Solution**: 
```bash
pip install -e .
```

#### C++ Extension Loading Failures

**Problem**: Tests fail with C++ extension import errors

**Solution**:
1. Ensure C++ extensions are built: `python setup.py build_ext --inplace`
2. Check that required system libraries are installed
3. Verify compiler compatibility

#### Test Failures After Code Changes

**Problem**: Tests that previously passed now fail

**Solution**:
1. Review your code changes for breaking changes
2. Check if test data or expected outputs need updating
3. Run tests in verbose mode (`-v`) to see detailed error messages
4. Check if reference files (`ref_*.csv`) need regeneration

#### Slow Test Execution

**Problem**: Tests take too long to run

**Solution**:
- Run specific test files instead of the full suite during development
- Use `pytest -x` to stop at first failure
- Use `pytest --lf` to run only failed tests from last run

### Getting Help

If you encounter issues:

1. Check the error messages carefully - they often contain helpful information
2. Review the test code to understand what's being tested
3. Compare your results with TraMineR outputs if testing distance measures
4. Check the main project documentation for API changes

## Additional Resources

- **Main Documentation**: See the main project README for usage examples
- **TraMineR Reference**: https://github.com/cran/TraMineR (for understanding expected behavior)
- **pytest Documentation**: https://docs.pytest.org/ (for advanced testing features)

## Contributing

When contributing to the test suite:

1. **Maintain test coverage**: Ensure new features have corresponding tests
2. **Keep tests fast**: Optimize slow tests or mark them appropriately
3. **Update documentation**: Update this README when adding new test categories
4. **Follow conventions**: Use existing test patterns and naming conventions

---

**Last Updated**: February 2026
**Author**: Yuqi Liang
**Maintainer**: Sequenzo Development Team
