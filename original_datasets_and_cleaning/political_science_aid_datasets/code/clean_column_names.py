"""
@Author  : Yuqi Liang 梁彧祺
@File    : clean_column_names.py
@Time    : 2026/2/2 15:15
@Desc    : 
Clean column names for political science aid datasets.

This script processes two CSV files:
1. political_science_aid_shock.csv - removes 'Shock_' prefix from year columns
2. political_science_donor_fragmentation.csv - removes 'HHI_state_' prefix from year columns

The cleaned files are saved to the cleaned_data directory without overwriting originals.

Uses the existing clean_time_columns_auto function from sequenzo.data_preprocessing.helpers.
"""

import pandas as pd
import sys
from pathlib import Path

# Add the sequenzo package to the path to import helpers
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sequenzo.data_preprocessing.helpers import clean_time_columns_auto

# Define paths
BASE_DIR = Path(__file__).parent
DATA_SOURCES_DIR = BASE_DIR / "data_sources"
CLEANED_DATA_DIR = BASE_DIR / "cleaned_data"

# Input file names
SHOCK_FILE = "political_science_aid_shock.csv"
FRAGMENTATION_FILE = "political_science_donor_fragmentation.csv"


def main():
    """
    Main function to process both datasets and save cleaned versions.
    Uses clean_time_columns_auto from sequenzo.data_preprocessing.helpers.
    """
    # Create cleaned_data directory if it doesn't exist
    CLEANED_DATA_DIR.mkdir(exist_ok=True)
    
    # Process political_science_aid_shock.csv
    print(f"Processing {SHOCK_FILE}...")
    shock_path = DATA_SOURCES_DIR / SHOCK_FILE
    if shock_path.exists():
        df_shock = pd.read_csv(shock_path)
        print(f"  Original columns: {list(df_shock.columns[:5])}... (showing first 5)")
        
        # Use clean_time_columns_auto with 'Shock_' prefix pattern
        df_shock_cleaned = clean_time_columns_auto(df_shock, prefix_patterns=['Shock_'])
        print(f"  Cleaned columns: {list(df_shock_cleaned.columns[:5])}... (showing first 5)")
        
        # Save cleaned file
        output_path = CLEANED_DATA_DIR / SHOCK_FILE
        df_shock_cleaned.to_csv(output_path, index=False)
        print(f"  Saved cleaned file to: {output_path}")
    else:
        print(f"  ERROR: File not found: {shock_path}")
    
    print()
    
    # Process political_science_donor_fragmentation.csv
    print(f"Processing {FRAGMENTATION_FILE}...")
    fragmentation_path = DATA_SOURCES_DIR / FRAGMENTATION_FILE
    if fragmentation_path.exists():
        df_fragmentation = pd.read_csv(fragmentation_path)
        print(f"  Original columns: {list(df_fragmentation.columns[:5])}... (showing first 5)")
        
        # Use clean_time_columns_auto with 'HHI_state_' prefix pattern
        df_fragmentation_cleaned = clean_time_columns_auto(df_fragmentation, prefix_patterns=['HHI_state_'])
        print(f"  Cleaned columns: {list(df_fragmentation_cleaned.columns[:5])}... (showing first 5)")
        
        # Save cleaned file
        output_path = CLEANED_DATA_DIR / FRAGMENTATION_FILE
        df_fragmentation_cleaned.to_csv(output_path, index=False)
        print(f"  Saved cleaned file to: {output_path}")
    else:
        print(f"  ERROR: File not found: {fragmentation_path}")
    
    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    main()
