# This file makes 'datasets' a Python package


def list_datasets():
    """List all available datasets in the `datasets` package."""
    from importlib.resources import files

    datasets_dir = files("sequenzo.datasets")
    datasets = [f.name.removesuffix(".csv") for f in datasets_dir.iterdir() if f.name.endswith(".csv")]
    return sorted(datasets)  # Sort alphabetically so related datasets are grouped together


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
    from importlib.resources import files

    available_datasets = list_datasets()  # Get the dynamic dataset list

    if name not in available_datasets:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available_datasets}")

    # Load the dataset from the package (files() is the modern replacement for open_text)
    with (files("sequenzo.datasets") / f"{name}.csv").open("r", encoding="utf-8") as f:
        return pd.read_csv(f)


# Key: Add this line to ensure load_dataset can be accessed externally
__all__ = ["load_dataset", "list_datasets"]