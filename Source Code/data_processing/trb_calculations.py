import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import ensure_dirs, PROCESSED_DATA_DIR

profiles = {
    'm_s': {'V_T': 0.00075, 'f': 12},
    'm_w': {'V_T': 0.00125, 'f': 20},
    'f_s': {'V_T': 0.00060, 'f': 14},
    'f_w': {'V_T': 0.00100, 'f': 21}
}
for p in profiles.values():
    p['VE'] = p['V_T'] * p['f']

def icrp_total_deposition(dp):
    ln_dp = np.log(dp)
    IF = 1 - 0.5 * (1 - 1 / (1 + 0.00076 * dp**2.8))
    DF_HA = IF * (1 / (1 + np.exp(6.84 + 1.183 * ln_dp)) + 1 / (1 + np.exp(0.924 - 1.885 * ln_dp)))
    DF_TB = (0.00352 / dp) * (np.exp(-0.234 * (ln_dp + 3.40)**2) + 63.9 * np.exp(-0.819 * (ln_dp - 1.61)**2))
    DF_AL = (0.0155 / dp) * (np.exp(-0.416 * (ln_dp + 2.84)**2) + 19.11 * np.exp(-0.482 * (ln_dp - 1.362)**2))
    return DF_HA + DF_TB + DF_AL

F_pm25 = icrp_total_deposition(2.5)
F_pm10 = icrp_total_deposition(10.0)

F_gas_scenarios = {
    'low':  {'so2':0.85,'no2':0.50,'o3':0.70,'co':0.80,'nh3':0.85},
    'mid':  {'so2':0.95,'no2':0.65,'o3':0.825,'co':0.875,'nh3':0.95},
    'high': {'so2':1.00,'no2':0.80,'o3':0.95,'co':0.95,'nh3':1.00}
}

who_ref = {'pm2.5':15,'pm10':45,'no2':25,'so2':40,'o3':100,'co':4000}
naaqs_ref = {'pm2.5':60,'pm10':100,'no2':80,'so2':80,'o3':100,'co':2000,'nh3':400}

def compute_ref_trb(profile, ref_dict, F_gas, include_nh3=False):
    VE = profile['VE']
    total = (ref_dict['pm2.5'] * VE * 60 * F_pm25 +
             ref_dict['pm10']  * VE * 60 * F_pm10 +
             ref_dict['no2']   * VE * 60 * F_gas['no2'] +
             ref_dict['so2']   * VE * 60 * F_gas['so2'] +
             ref_dict['o3']    * VE * 60 * F_gas['o3'] +
             ref_dict['co']    * VE * 60 * F_gas['co'])
    if include_nh3 and 'nh3' in ref_dict:
        total += ref_dict['nh3'] * VE * 60 * F_gas['nh3']
    return total

ref_who_mid = {k: compute_ref_trb(p, who_ref, F_gas_scenarios['mid'], False) for k,p in profiles.items()}
ref_naaqs_mid = {k: compute_ref_trb(p, naaqs_ref, F_gas_scenarios['mid'], True) for k,p in profiles.items()}

def add_trb(input_path, output_path):
    df = pd.read_csv(input_path)
    for prof_name, prof_param in profiles.items():
        VE = prof_param['VE']
        df[f'TRB_WHO_ref_{prof_name}'] = ref_who_mid[prof_name]
        df[f'TRB_NAAQS_ref_{prof_name}'] = ref_naaqs_mid[prof_name]
        for scenario, F_gas in F_gas_scenarios.items():
            actual_col = f'TRB_actual_{prof_name}_{scenario}'
            df[actual_col] = (df['pm2.5'] * VE * 60 * F_pm25 +
                              df['pm10']  * VE * 60 * F_pm10 +
                              df['no2']   * VE * 60 * F_gas['no2'] +
                              df['so2']   * VE * 60 * F_gas['so2'] +
                              df['o3']    * VE * 60 * F_gas['o3'] +
                              df['co']    * 1000 * VE * 60 * F_gas['co'] +
                              df['nh3']   * VE * 60 * F_gas['nh3'])
            df[f'TRBI_WHO_{prof_name}_{scenario}'] = df[actual_col] / ref_who_mid[prof_name]
            df[f'TRBI_NAAQS_{prof_name}_{scenario}'] = df[actual_col] / ref_naaqs_mid[prof_name]
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].round(2)
    df.to_csv(output_path, index=False)
    print(f"Foundational dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROCESSED_DATA_DIR / "aqi.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DATA_DIR / "foundational_dataset.csv"))
    args = parser.parse_args()
    ensure_dirs()
    add_trb(args.input, args.output)