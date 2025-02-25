# This file makes 'datasets' a Python package

import os
import importlib.resources as pkg_resources
import pandas as pd

def list_datasets():
    """List all available datasets in the `datasets` package."""
    with pkg_resources.path("sequenzo.datasets", "__init__.py") as datasets_path:
        datasets_dir = datasets_path.parent  # 获取 datasets 目录路径
        return [file.stem for file in datasets_dir.iterdir() if file.suffix == ".csv"]

def load_dataset(name):
    """
    Load a built-in dataset from the sequenzo package dynamically.

    Parameters:
        name (str): The name of the dataset (without `.csv`).

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    available_datasets = list_datasets()  # 获取动态数据集列表

    if name not in available_datasets:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {available_datasets}")

    # Load the dataset from the package
    with pkg_resources.open_text("sequenzo.datasets", f"{name}.csv") as f:
        return pd.read_csv(f)

# 🚀 **关键：添加这一行，确保 load_dataset 可以被外部访问**
__all__ = ["load_dataset", "list_datasets"]


