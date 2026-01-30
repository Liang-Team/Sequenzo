"""
@Author  : Yuqi Liang 梁彧祺
@File    : split_and_preprocess_pairfam.py
@Time    : 2026/1/30 09:34
@Desc    : 
    Split and preprocess Pairfam MultiChannel data.

    This script splits the MultiChannel.csv file into two separate datasets:
    1. pairfam_family: Contains family state sequences (family1...family264 -> 1...264)
    2. pairfam_activity: Contains activity state sequences (activity1...activity264 -> 1...264)

    Data preprocessing:
    - Converts column names like 'family11' to '11', 'activity1' to '1', etc.
    - Uses clean_time_columns_auto() function to automatically extract numbers from column names
"""

import pandas as pd
import re
import os


def clean_time_columns_auto(df: pd.DataFrame, 
                             prefix_patterns: list = None) -> pd.DataFrame:
    """
    Automatically clean column names by extracting numbers from them.
    
    This function scans a DataFrame, identifies columns with names containing numbers
    (e.g., state1, wave2, year2023, family11), and simplifies these names to just
    the numbers they contain (becoming 1, 2, 2023, 11).
    
    This feature is particularly useful when processing time-series or panel data,
    as it allows for quick standardization of column names that represent different
    points in time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns that may contain numbers in their names
    prefix_patterns : list, optional
        List of prefixes to match (e.g., ['family', 'activity', 'state']).
        If provided, only columns starting with these prefixes will be cleaned.
        If None, all columns with numbers will be cleaned.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned column names (numbers extracted from column names)
    
    Examples:
    ---------
    >>> df = pd.DataFrame({'id': [1, 2], 'family1': [1, 2], 'family11': [3, 4]})
    >>> cleaned = clean_time_columns_auto(df, prefix_patterns=['family'])
    >>> cleaned.columns.tolist()
    ['id', '1', '11']
    
    >>> df = pd.DataFrame({'id': [1], 'state1': [1], 'state264': [2]})
    >>> cleaned = clean_time_columns_auto(df, prefix_patterns=['state'])
    >>> cleaned.columns.tolist()
    ['id', '1', '264']
    """
    df_cleaned = df.copy()
    new_columns = {}
    
    for col in df.columns:
        # Check if we should process this column
        should_process = False
        
        if prefix_patterns is None:
            # Process all columns with numbers
            should_process = bool(re.search(r'\d+', col))
        else:
            # Only process columns that start with one of the specified prefixes
            should_process = any(col.startswith(prefix) for prefix in prefix_patterns)
        
        if should_process:
            # Extract all numbers from the column name
            # This regex finds all sequences of digits in the column name
            numbers = re.findall(r'\d+', col)
            
            if numbers:
                # If numbers are found, use the last (or only) number sequence
                # This handles cases like 'family11' -> '11', 'state1' -> '1'
                # For cases like 'year2023', it will extract '2023'
                extracted_number = numbers[-1]  # Take the last number found
                
                # Only rename if the column name actually contains non-numeric characters
                # This prevents renaming columns that are already just numbers
                if re.search(r'[a-zA-Z]', col):
                    new_columns[col] = extracted_number
                else:
                    # Column is already numeric, keep it as is
                    new_columns[col] = col
            else:
                # No numbers found, keep original column name
                new_columns[col] = col
        else:
            # Don't process this column, keep original name
            new_columns[col] = col
    
    # Rename columns
    df_cleaned = df_cleaned.rename(columns=new_columns)
    
    return df_cleaned


def split_pairfam_data(input_csv_path: str, 
                      output_dir: str = None) -> tuple:
    """
    Split MultiChannel.csv into pairfam_family and pairfam_activity datasets.
    
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
    family_output_path = os.path.join(output_dir, 'pairfam_family.csv')
    activity_output_path = os.path.join(output_dir, 'pairfam_activity.csv')
    
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
        'pairfam_and_little_green_book',
        'data_sources',
        'MultiChannel.csv'
    )
    
    output_dir = os.path.join(
        project_root,
        'pairfam_and_little_green_book',
        'data_sources'
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
        print(f"✓ Created pairfam_family.csv with {len(pairfam_family.columns)} columns")
        print(f"✓ Created pairfam_activity.csv with {len(pairfam_activity.columns)} columns")
        print(f"✓ Column names cleaned (e.g., family11 -> 11, activity1 -> 1)")
        print("="*60)
