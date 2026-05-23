import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import ensure_dirs, RAW_DATASET_PATH, PROCESSED_DATA_DIR, RAW_POLLUTANTS

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Divide CO by 1000 (convert to mg/m³)
    df['co'] = df['co'] / 1000
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Impute missing values per city with ffill + bfill (sorted by datetime)
    unique_cities = df['city'].unique()
    for city in unique_cities:
        city_mask = df['city'] == city
        city_indices = df[city_mask].index
        city_df = df.loc[city_indices].copy()
        city_df.sort_values('datetime', inplace=True)
        for col in RAW_POLLUTANTS:
            if city_df[col].isnull().any():
                city_df[col] = city_df[col].ffill().bfill()
        df.loc[city_indices, RAW_POLLUTANTS] = city_df[RAW_POLLUTANTS]
    
    # Cap pm2.5 where it exceeds pm10
    condition = df['pm2.5'] > df['pm10']
    df.loc[condition, 'pm2.5'] = df.loc[condition, 'pm10']
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_DATASET_PATH), help="Raw input CSV")
    parser.add_argument("--output", default=str(PROCESSED_DATA_DIR / "cleaned.csv"), help="Output CSV")
    args = parser.parse_args()
    ensure_dirs()
    clean_data(args.input, args.output)