# -*- coding: utf-8 -*-
"""Session 1: Information Theoretic Audit – Portable version"""

import sys
import warnings
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy.stats import rankdata
import infomeasure as im

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import (ensure_dirs, DATASET_PATH, RESULTS_DIR, FIGURES_DIR,
                         CITIES, RAW_POLLUTANTS, AQI_CATEGORIES, ALL_TARGETS,
                         TIER1_TARGETS, REPR_TARGET, SUBINDEX_COLS, RANDOM_SEED)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# Output directories
session_dir = RESULTS_DIR / "session_1"
session_dir.mkdir(parents=True, exist_ok=True)
fig_dir = FIGURES_DIR / "session1"
fig_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df = df.sort_values(['city', 'datetime']).reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"Cities: {df['city'].unique()}")

# Helper functions
def rank_transform(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return rankdata(arr).astype(float)
    return np.column_stack([rankdata(arr[:, j]) for j in range(arr.shape[1])])

def ksg_mi(x, y, k=4, seed=RANDOM_SEED) -> float:
    np.random.seed(seed)
    x_r = rank_transform(np.asarray(x))
    y_r = rank_transform(np.asarray(y))
    est = im.estimator(
        x_r, y_r,
        measure='mutual_information',
        approach='metric',
        ksg_id=1,
        k=k,
        minkowski_p=np.inf,
        noise_level=1e-10,
        normalize=False
    )
    mi = est.result()
    return float(max(mi, 0.0))

def ksg_mi_with_test(x, y, n_perm=200, k=4, seed=RANDOM_SEED):
    np.random.seed(seed)
    x_r = rank_transform(np.asarray(x))
    y_r = rank_transform(np.asarray(y))
    est = im.estimator(
        x_r, y_r,
        measure='mutual_information',
        approach='metric',
        ksg_id=1,
        k=k,
        minkowski_p=np.inf,
        noise_level=1e-10,
        normalize=False
    )
    mi_val = float(max(est.result(), 0.0))
    test = est.statistical_test(n_tests=n_perm, method='permutation_test')
    return mi_val, test.p_value, test.t_score

def save_csv_with_meta(df_out, filepath, description, col_desc):
    filepath = Path(filepath)
    df_out.to_csv(filepath, index=False)
    meta = {
        'created_at': datetime.datetime.now().isoformat(),
        'description': description,
        'column_descriptions': col_desc,
        'random_seed': RANDOM_SEED
    }
    with open(filepath.with_suffix('.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {filepath.name}")

# ----------------------------------------------------------------------
# 1. I(X_raw ; TRB)
print("\n=== 1. I(X_raw_pollutants ; TRB) ===")
mi_joint = {}
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi']).copy()
    X_raw = dfc[RAW_POLLUTANTS].values
    mi_joint[city] = {}
    for target in ALL_TARGETS:
        mi_val = ksg_mi(X_raw, dfc[target].values)
        mi_joint[city][target] = mi_val
        print(f"{city} {target}: {mi_val:.4f} nats")

# 2. I(AQI_cont ; TRB)
print("\n=== 2. I(AQI_continuous ; TRB) ===")
mi_aqi_cont = {}
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi'] + ALL_TARGETS)
    aqi_vals = dfc['aqi'].values.reshape(-1,1)
    mi_aqi_cont[city] = {}
    for target in ALL_TARGETS:
        mi_val = ksg_mi(aqi_vals, dfc[target].values)
        mi_aqi_cont[city][target] = mi_val
        print(f"{city} {target}: {mi_val:.4f} nats")

# 3. I(AQI_category_onehot ; TRB)
print("\n=== 3. I(AQI_category_onehot ; TRB) ===")
mi_aqi_cat = {}
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi_category'] + ALL_TARGETS)
    cat_ohe = pd.get_dummies(dfc['aqi_category'], columns=AQI_CATEGORIES, dtype=float).values
    mi_aqi_cat[city] = {}
    for target in ALL_TARGETS:
        mi_val = ksg_mi(cat_ohe, dfc[target].values)
        mi_aqi_cat[city][target] = mi_val
        print(f"{city} {target}: {mi_val:.4f} nats")

# Compression ratios
rows = []
for city in CITIES:
    for target in ALL_TARGETS:
        i_x = mi_joint[city][target]
        i_cont = mi_aqi_cont[city][target]
        i_cat = mi_aqi_cat[city][target]
        rows.append({
            'city': city, 'target': target,
            'I_X_TRB': i_x, 'I_AQI_cont_TRB': i_cont, 'I_AQI_cat_TRB': i_cat,
            'ratio_cont': i_cont / i_x if i_x > 0 else np.nan,
            'ratio_cat': i_cat / i_x if i_x > 0 else np.nan,
        })
df_ratios = pd.DataFrame(rows)
save_csv_with_meta(df_ratios, session_dir / 'compression_ratios.csv',
                   'MI compression ratios', {})

# 4. Univariate MI
print("\n=== 4. Univariate MI ===")
mi_univar = {}
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi']).copy()
    mi_univar[city] = {}
    for poll in RAW_POLLUTANTS:
        mi_univar[city][poll] = {}
        x_vals = dfc[poll].values
        for target in ALL_TARGETS:
            mi_val = ksg_mi(x_vals, dfc[target].values)
            mi_univar[city][poll][target] = mi_val
        print(f"{city} {poll}: {mi_univar[city][poll][REPR_TARGET]:.4f} nats")

# 5. Interaction information for all pairs
print("\n=== 5. Interaction information ===")
mi_pairs = {}
poll_pairs = list(combinations(RAW_POLLUTANTS, 2))
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi']).copy()
    mi_pairs[city] = {}
    for p1, p2 in poll_pairs:
        key = f"{p1}__{p2}"
        mi_pairs[city][key] = {}
        X_pair = dfc[[p1, p2]].values
        for target in ALL_TARGETS:
            i_joint = ksg_mi(X_pair, dfc[target].values)
            i_p1 = mi_univar[city][p1][target]
            i_p2 = mi_univar[city][p2][target]
            interaction = i_joint - i_p1 - i_p2
            mi_pairs[city][key][target] = {
                'I_joint': i_joint, 'I_p1': i_p1, 'I_p2': i_p2,
                'interaction': interaction,
                'type': 'synergy' if interaction > 0 else 'redundancy'
            }
        ii = mi_pairs[city][key][REPR_TARGET]['interaction']
        print(f"{city} {key}: II={ii:+.4f}")

# Save interaction info
rows_ii = []
for city in CITIES:
    for key, tgt_data in mi_pairs[city].items():
        p1, p2 = key.split('__')
        for tgt, vals in tgt_data.items():
            rows_ii.append({'city': city, 'p1': p1, 'p2': p2, 'target': tgt, **vals})
df_ii = pd.DataFrame(rows_ii)
save_csv_with_meta(df_ii, session_dir / 'interaction_info.csv', 'Interaction information', {})

# 6. Stratified interaction by sub-index gap
print("\n=== 6. Stratified interaction (low-gap vs high-gap) ===")
GAP_THRESH = 20
def compute_gap(row):
    vals = [row[c] for c in SUBINDEX_COLS if pd.notna(row[c])]
    if len(vals) < 2:
        return 0.0
    sv = sorted(vals, reverse=True)
    return sv[0] - sv[1]

stratified_ii = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=RAW_POLLUTANTS + SUBINDEX_COLS).copy()
    dfc['gap'] = dfc.apply(compute_gap, axis=1)
    df_low = dfc[dfc['gap'] < GAP_THRESH]
    df_high = dfc[dfc['gap'] >= GAP_THRESH]
    for stratum, stratum_df in [('low_gap', df_low), ('high_gap', df_high)]:
        if len(stratum_df) < 100:
            continue
        for p1, p2 in poll_pairs:
            X_pair = stratum_df[[p1, p2]].values
            for target in TIER1_TARGETS:
                i_joint = ksg_mi(X_pair, stratum_df[target].values)
                i_p1 = ksg_mi(stratum_df[p1].values, stratum_df[target].values)
                i_p2 = ksg_mi(stratum_df[p2].values, stratum_df[target].values)
                interaction = i_joint - i_p1 - i_p2
                stratified_ii.append({
                    'city': city, 'stratum': stratum, 'p1': p1, 'p2': p2,
                    'target': target, 'I_joint': i_joint, 'I_p1': i_p1, 'I_p2': i_p2,
                    'interaction': interaction
                })
df_strat = pd.DataFrame(stratified_ii)
save_csv_with_meta(df_strat, session_dir / 'stratified_interaction_info.csv',
                   'Stratified interaction information', {})

# 7. Permutation tests for H1
print("\n=== 7. Permutation tests (multi-pollutant vs max-subindex) ===")
sig_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=RAW_POLLUTANTS + ['aqi'])
    max_sub = dfc['aqi'].values.reshape(-1,1)
    X_all = dfc[RAW_POLLUTANTS].values
    for target in TIER1_TARGETS:
        y_vals = dfc[target].values
        mi_all, p_all, _ = ksg_mi_with_test(X_all, y_vals, n_perm=200)
        mi_max, p_max, _ = ksg_mi_with_test(max_sub, y_vals, n_perm=200)
        sig_results.append({
            'city': city, 'target': target,
            'MI_all': mi_all, 'p_all': p_all,
            'MI_max': mi_max, 'p_max': p_max,
            'delta_MI': mi_all - mi_max,
            'H1_supported': (mi_all > mi_max) and (p_all < 0.05)
        })
df_sig = pd.DataFrame(sig_results)
save_csv_with_meta(df_sig, session_dir / 'h1_significance.csv', 'H1 permutation tests', {})

# 8. Bootstrap CI for compression ratios
print("\n=== 8. Bootstrap CI for compression ratios (200 resamples) ===")
N_BOOT = 200
ci_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=RAW_POLLUTANTS + TIER1_TARGETS + ['aqi']).reset_index(drop=True)
    X_raw = dfc[RAW_POLLUTANTS].values
    aqi_v = dfc['aqi'].values.reshape(-1,1)
    n_total = len(dfc)
    for target in TIER1_TARGETS:
        y_vals = dfc[target].values
        ratios = []
        for _ in range(N_BOOT):
            idx = np.random.choice(n_total, size=n_total, replace=True)
            i_x = ksg_mi(X_raw[idx], y_vals[idx])
            i_a = ksg_mi(aqi_v[idx], y_vals[idx])
            if i_x > 0:
                ratios.append(i_a / i_x)
        if ratios:
            ci_results.append({
                'city': city, 'target': target,
                'ratio_mean': np.mean(ratios),
                'ci_lower': np.percentile(ratios, 2.5),
                'ci_upper': np.percentile(ratios, 97.5)
            })
df_ci = pd.DataFrame(ci_results)
save_csv_with_meta(df_ci, session_dir / 'bootstrap_ci_compression_ratios.csv', 'Bootstrap CI', {})

# 9. Figures: heatmaps and bar charts
print("\n=== 9. Generating figures ===")

# Compression ratio heatmap
for metric, title, suffix in [('ratio_cont', 'Continuous AQI', 'cont'),
                               ('ratio_cat', 'Categorical AQI', 'cat')]:
    pivot = df_ratios.pivot_table(index='city', columns='target', values=metric)
    ordered_cols = [c for c in ALL_TARGETS if c in pivot.columns]
    pivot = pivot[ordered_cols]
    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
    ax.set_title(f'Compression Ratio — {title}', fontsize=14)
    plt.tight_layout()
    fig.savefig(fig_dir / f'heatmap_ratio_{suffix}.png', dpi=300)
    fig.savefig(fig_dir / f'heatmap_ratio_{suffix}.svg', dpi=300)
    plt.close()

# Interaction information heatmap per city
for city in CITIES:
    mat = np.zeros((7,7))
    labels = RAW_POLLUTANTS
    for i, p1 in enumerate(labels):
        for j, p2 in enumerate(labels):
            if i < j:
                key = f"{p1}__{p2}"
                ii = mi_pairs[city].get(key, {}).get(REPR_TARGET, {}).get('interaction', 0)
                mat[i,j] = ii
                mat[j,i] = ii
    fig, ax = plt.subplots(figsize=(8,7))
    lim = np.nanmax(np.abs(mat)) + 0.01
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', vmin=-lim, vmax=lim,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'{city} — Interaction Information (nats)\nTarget: {REPR_TARGET}')
    plt.tight_layout()
    fig.savefig(fig_dir / f'interaction_info_{city.lower()}.png', dpi=300)
    fig.savefig(fig_dir / f'interaction_info_{city.lower()}.svg', dpi=300)
    plt.close()

# Univariate MI bar chart
rows_u = []
for city in CITIES:
    for poll in RAW_POLLUTANTS:
        rows_u.append({'city': city, 'pollutant': poll, 'MI': mi_univar[city][poll][REPR_TARGET]})
df_univar = pd.DataFrame(rows_u)
fig, axes = plt.subplots(1, 3, figsize=(16,5))
for ax, city in zip(axes, CITIES):
    sub = df_univar[df_univar['city']==city].sort_values('MI', ascending=False)
    ax.bar(sub['pollutant'], sub['MI'], color=sns.color_palette('muted',7))
    ax.set_title(city)
    ax.set_ylabel('MI (nats)' if city==CITIES[0] else '')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle(f'Univariate MI: I(pollutant ; {REPR_TARGET})')
plt.tight_layout()
fig.savefig(fig_dir / 'univariate_mi_bar.png', dpi=300)
fig.savefig(fig_dir / 'univariate_mi_bar.svg', dpi=300)
plt.close()

print("\n=== Session 1 complete ===")
print(f"Results saved to {session_dir}")
print(f"Figures saved to {fig_dir}")