# This file makes 'datasets' a Python package


def list_datasets():
    """List all available datasets in the `datasets` package."""
    # Delay imports to avoid circular dependency issues during installation
    import importlib.resources as pkg_resources

    with pkg_resources.path("sequenzo.datasets", "__init__.py") as datasets_path:
        datasets_dir = datasets_path.parent  # Get the datasets directory path
        return [file.stem for file in datasets_dir.iterdir() if file.suffix == ".csv"]


def load_dataset(name):
    """
    Load a built-in dataset from the sequenzo package dynamically.

    Parameters:
        name (str): The name of the dataset (without `.csv`).

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    # Import pandas only when the function is called, not when the module is loaded
    import pandas as pd
    import os
    # Import resources management module
    import importlib.resources as pkg_resources

    available_datasets = list_datasets()  # Get the dynamic dataset list

    if name not in available_datasets:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available_datasets}")

    # Load the dataset from the package
    with pkg_resources.open_text("sequenzo.datasets", f"{name}.csv") as f:
        return pd.read_csv(f)


# Key: Add this line to ensure load_dataset can be accessed externally
__all__ = ["load_dataset", "list_datasets"]