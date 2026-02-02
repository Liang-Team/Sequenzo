"""
@Author  : Yuqi Liang 梁彧祺
@File    : split_and_preprocess_pairfam.py
@Time    : 2026/1/30 09:34
@Desc    : 
    Split and preprocess Pairfam MultiChannel data.

    This script splits the MultiChannel.csv file into two separate datasets:
    1. pairfam_family_by_month.csv: Family state sequences (family1...family264 -> 1...264)
    2. pairfam_activity_by_month.csv: Activity state sequences (activity1...activity264 -> 1...264)

    Data preprocessing:
    - Converts column names like 'family11' to '11', 'activity1' to '1', etc.
    - Uses clean_time_columns_auto() function from sequenzo.data_preprocessing.helpers
      to automatically extract numbers from column names
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add the sequenzo package to the path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from sequenzo.data_preprocessing.helpers import clean_time_columns_auto


def split_pairfam_data(input_csv_path: str, 
                      output_dir: str = None) -> tuple:
    """
    Split MultiChannel.csv into pairfam_family_by_month and pairfam_activity_by_month datasets.
    
    Parameters:
    -----------
    input_csv_path : str
        Path to the input MultiChannel.csv file
    output_dir : str, optional
        Directory to save output files. If None, saves in the same directory as input file.
        
    Returns:
    --------
    tuple
        Tuple containing (pairfam_family_df, pairfam_activity_df)
    """
    # Read the input CSV file
    print(f"Reading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Identify metadata columns (non-family, non-activity columns)
    # These are columns that don't start with 'family' or 'activity'
    metadata_cols = [col for col in df.columns 
                     if not col.startswith('family') and not col.startswith('activity')]
    
    # Identify family columns (family1 to family264)
    family_cols = [col for col in df.columns if col.startswith('family')]
    
    # Identify activity columns (activity1 to activity264)
    activity_cols = [col for col in df.columns if col.startswith('activity')]
    
    print(f"\nMetadata columns: {len(metadata_cols)}")
    print(f"Family columns: {len(family_cols)}")
    print(f"Activity columns: {len(activity_cols)}")
    
    # Create pairfam_family dataset: metadata + family columns
    pairfam_family = df[metadata_cols + family_cols].copy()
    
    # Create pairfam_activity dataset: metadata + activity columns
    pairfam_activity = df[metadata_cols + activity_cols].copy()
    
    # Apply clean_time_columns_auto() to clean column names
    # Only clean columns that start with 'family' or 'activity'
    print("\nCleaning column names for pairfam_family...")
    pairfam_family = clean_time_columns_auto(pairfam_family, prefix_patterns=['family'])
    
    print("Cleaning column names for pairfam_activity...")
    pairfam_activity = clean_time_columns_auto(pairfam_activity, prefix_patterns=['activity'])
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed datasets
    family_output_path = os.path.join(output_dir, 'pairfam_family_by_month.csv')
    activity_output_path = os.path.join(output_dir, 'pairfam_activity_by_month.csv')
    
    print(f"\nSaving pairfam_family to: {family_output_path}")
    pairfam_family.to_csv(family_output_path, index=False)
    
    print(f"Saving pairfam_activity to: {activity_output_path}")
    pairfam_activity.to_csv(activity_output_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"pairfam_family shape: {pairfam_family.shape}")
    print(f"pairfam_activity shape: {pairfam_activity.shape}")
    
    # Show sample of cleaned column names
    print(f"\nSample cleaned column names in pairfam_family:")
    print(f"  First few columns: {list(pairfam_family.columns[:10])}")
    print(f"  Last few columns: {list(pairfam_family.columns[-10:])}")
    
    print(f"\nSample cleaned column names in pairfam_activity:")
    print(f"  First few columns: {list(pairfam_activity.columns[:10])}")
    print(f"  Last few columns: {list(pairfam_activity.columns[-10:])}")
    
    return pairfam_family, pairfam_activity


if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_csv_path = os.path.join(
        project_root,
        'data_sources',
        'month_level',
        'MultiChannel.csv'
    )
    
    output_dir = os.path.join(
        project_root,
        'data_sources',
        'month_level'
    )
    
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at {input_csv_path}")
        print("Please check the file path.")
    else:
        # Run the splitting and preprocessing
        pairfam_family, pairfam_activity = split_pairfam_data(
            input_csv_path=input_csv_path,
            output_dir=output_dir
        )
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"✓ Successfully split MultiChannel.csv")
        print(f"✓ Created pairfam_family_by_month.csv with {len(pairfam_family.columns)} columns")
        print(f"✓ Created pairfam_activity_by_month.csv with {len(pairfam_activity.columns)} columns")
        print(f"✓ Column names cleaned (e.g., family11 -> 11, activity1 -> 1)")
        print("="*60)
