import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import ensure_dirs, PROCESSED_DATA_DIR

# Breakpoints definition (same as original)
breakpoints = {
    'pm2.5': {'categories': [(0,30,0,50),(31,60,51,100),(61,90,101,200),(91,120,201,300),(121,250,301,400)], 'severe':(250,0.76923)},
    'pm10': {'categories': [(0,50,0,50),(51,100,51,100),(101,250,101,200),(251,350,201,300),(351,430,301,400)], 'severe':(430,1.25)},
    'no2': {'categories': [(0,40,0,50),(41,80,51,100),(81,180,101,200),(181,280,201,300),(281,400,301,400)], 'severe':(400,0.83333)},
    'so2': {'categories': [(0,40,0,50),(41,80,51,100),(81,380,101,200),(381,800,201,300),(801,1600,301,400)], 'severe':(1600,0.125)},
    'co': {'categories': [(0,1.0,0,50),(1.1,2.0,51,100),(2.1,10,101,200),(10,17,201,300),(17,34,301,400)], 'severe':(34,5.88235)},
    'o3': {'categories': [(0,50,0,50),(51,100,51,100),(101,168,101,200),(169,208,201,300),(209,748,301,400)], 'severe':(748,0.18553)},
    'nh3': {'categories': [(0,200,0,50),(201,400,51,100),(401,800,101,200),(801,1200,201,300),(1200,1800,301,400)], 'severe':(1800,0.16667)}
}

def compute_subindex(pollutant, conc):
    if pd.isna(conc): return np.nan
    bp = breakpoints[pollutant]
    for c_low, c_high, i_low, i_high in bp['categories']:
        if conc <= c_high:
            return i_low + (i_high - i_low) * (conc - c_low) / (c_high - c_low)
    c_high_vp, slope = bp['severe']
    return 400 + slope * (conc - c_high_vp)

def compute_aqi(input_path, output_path):
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # 24h rolling averages
    for poll in ['pm2.5', 'pm10', 'no2', 'so2', 'nh3']:
        df[f'{poll}_24hr'] = df.groupby('city')[poll].rolling(window=24, min_periods=16).mean().reset_index(level=0, drop=True)
    # 8h rolling averages
    for poll in ['co', 'o3']:
        df[f'{poll}_8hr'] = df.groupby('city')[poll].rolling(window=8, min_periods=8).mean().reset_index(level=0, drop=True)
    
    # Subindices
    avg_cols = {'pm2.5':'pm2.5_24hr','pm10':'pm10_24hr','no2':'no2_24hr','so2':'so2_24hr','nh3':'nh3_24hr','co':'co_8hr','o3':'o3_8hr'}
    for poll, avg_col in avg_cols.items():
        df[f'{poll}_subindex'] = df.apply(lambda row: compute_subindex(poll, row[avg_col]), axis=1)
    
    # AQI
    def aqi_and_category(row):
        sub_cols = [f'{p}_subindex' for p in avg_cols.keys()]
        valid = [row[col] for col in sub_cols if pd.notna(row[col])]
        if len(valid) >= 3 and (pd.notna(row['pm2.5_subindex']) or pd.notna(row['pm10_subindex'])):
            aqi = max(valid)
            if aqi <= 50: cat = "Good"
            elif aqi <= 100: cat = "Satisfactory"
            elif aqi <= 200: cat = "Moderately Polluted"
            elif aqi <= 300: cat = "Poor"
            elif aqi <= 400: cat = "Very Poor"
            else: cat = "Severe"
            return aqi, cat
        return np.nan, np.nan
    df[['aqi','aqi_category']] = df.apply(lambda row: pd.Series(aqi_and_category(row)), axis=1)
    
    # Round and save
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(2)
    df['aqi'] = np.ceil(df['aqi']).astype('Int64')
    df.to_csv(output_path, index=False)
    print(f"AQI data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROCESSED_DATA_DIR / "cleaned.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DATA_DIR / "aqi.csv"))
    args = parser.parse_args()
    ensure_dirs()
    compute_aqi(args.input, args.output)