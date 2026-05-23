#!/usr/bin/env python3
"""
Run the full data processing pipeline:
1. Clean raw data
2. Compute AQI and subindices
3. Compute TRB and TRBI
All outputs are placed in the directories defined in config.json.
"""

import subprocess
import sys
from pathlib import Path

# Ensure utils is in path
sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import ensure_dirs, RAW_DATASET_PATH, PROCESSED_DATA_DIR

def main():
    ensure_dirs()
    
    # Check if raw data exists
    if not RAW_DATASET_PATH.exists():
        print(f"ERROR: Raw data file not found at {RAW_DATASET_PATH}")
        print("Please place your raw CSV file (named 'city_pollutants_raw.csv') in the 'data/raw/' folder.")
        sys.exit(1)
    
    # Step 1: Cleaning
    print("Step 1: Cleaning raw data...")
    subprocess.run([sys.executable, "data_processing/data_cleaning.py",
                    "--input", str(RAW_DATASET_PATH),
                    "--output", str(PROCESSED_DATA_DIR / "cleaned.csv")])
    
    # Step 2: AQI calculation
    print("Step 2: Computing AQI...")
    subprocess.run([sys.executable, "data_processing/aqi_calculations.py",
                    "--input", str(PROCESSED_DATA_DIR / "cleaned.csv"),
                    "--output", str(PROCESSED_DATA_DIR / "aqi.csv")])
    
    # Step 3: TRB calculation
    print("Step 3: Computing TRB and TRBI...")
    subprocess.run([sys.executable, "data_processing/trb_calculations.py",
                    "--input", str(PROCESSED_DATA_DIR / "aqi.csv"),
                    "--output", str(PROCESSED_DATA_DIR / "foundational_dataset.csv")])
    
    print("\nPipeline completed successfully.")
    print(f"Final dataset: {PROCESSED_DATA_DIR / 'foundational_dataset.csv'}")

if __name__ == "__main__":
    main()