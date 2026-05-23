# -*- coding: utf-8 -*-
"""Official vs TRB Breakpoints – Portable version"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import ensure_dirs, DATASET_PATH, RESULTS_DIR, FIGURES_DIR, REPR_TARGET, RAW_POLLUTANTS, RANDOM_SEED

warnings.filterwarnings('ignore')

# Output paths
results_dir = RESULTS_DIR / "session3"
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir = FIGURES_DIR / "session3"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df_clean = df.dropna(subset=['aqi', REPR_TARGET]).copy()
print(f"Total rows (all cities combined): {len(df_clean)}")
print(f"AQI range: {df_clean['aqi'].min():.1f} – {df_clean['aqi'].max():.1f}")

# Sort by AQI
df_sorted = df_clean.sort_values('aqi').reset_index(drop=True)
X = df_sorted['aqi'].values.reshape(-1, 1)
y = df_sorted[REPR_TARGET].values

# Method 1: Decision Tree Stumps
print("\n--- Method 1: Decision Tree (max_leaf_nodes=6) ---")
dt = DecisionTreeRegressor(max_leaf_nodes=6, random_state=RANDOM_SEED)
dt.fit(X, y)
thresh_tree = []
for i in range(dt.tree_.node_count):
    if dt.tree_.feature[i] != -2:
        thresh_tree.append(dt.tree_.threshold[i])
thresh_tree = sorted(set(thresh_tree))[:5]
print("5 breakpoints:", [round(t, 1) for t in thresh_tree])

# Method 2: GMM on 7 pollutants
print("\n--- Method 2: GMM on 7 pollutants ---")
pollutants = RAW_POLLUTANTS
X_poll = df_clean[pollutants].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poll)
gmm = GaussianMixture(n_components=6, random_state=RANDOM_SEED, n_init=5)
labels = gmm.fit_predict(X_scaled)
valid_idx = df_clean[pollutants].dropna().index
df_gmm = df_clean.loc[valid_idx].copy()
df_gmm['cluster'] = labels
medians = df_gmm.groupby('cluster')['aqi'].median().sort_values()
thresh_gmm = [(medians.iloc[i] + medians.iloc[i+1])/2 for i in range(5)]
thresh_gmm.sort()
print("5 breakpoints:", [round(t, 1) for t in thresh_gmm])

# Method 3: Mean of DT and GMM
print("\n--- Method 3: Mean of Decision Tree and GMM breakpoints ---")
thresh_mean = [(dt_bp + gmm_bp) / 2 for dt_bp, gmm_bp in zip(thresh_tree, thresh_gmm)]
thresh_mean.sort()
print("5 breakpoints:", [round(t, 1) for t in thresh_mean])

# Save results
results_df = pd.DataFrame({
    'Method': ['Decision Tree', 'GMM', 'Mean (DT+GMM)'],
    'Breakpoint_1': [thresh_tree[0], thresh_gmm[0], thresh_mean[0]],
    'Breakpoint_2': [thresh_tree[1], thresh_gmm[1], thresh_mean[1]],
    'Breakpoint_3': [thresh_tree[2], thresh_gmm[2], thresh_mean[2]],
    'Breakpoint_4': [thresh_tree[3], thresh_gmm[3], thresh_mean[3]],
    'Breakpoint_5': [thresh_tree[4], thresh_gmm[4], thresh_mean[4]],
})
results_df.to_csv(results_dir / 'aqi_5_breakpoints_pooled.csv', index=False)
print(f"\nResults saved to {results_dir / 'aqi_5_breakpoints_pooled.csv'}")

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_plot = df_clean[df_clean['aqi'] <= 500].copy()
bin_edges = np.arange(0, 510, 10)
bin_centers = bin_edges[:-1] + 5
df_plot['aqi_bin'] = pd.cut(df_plot['aqi'], bins=bin_edges, labels=bin_centers).astype(float)
bin_stats = df_plot.groupby('aqi_bin')[REPR_TARGET].agg(['mean', 'std', 'count']).reset_index()
bin_stats['sem'] = bin_stats['std'] / np.sqrt(bin_stats['count'])
bin_stats['ci95'] = bin_stats['sem'] * stats.t.ppf(0.975, bin_stats['count'] - 1)

breakpoints = [bp for bp in thresh_mean if bp <= 500]
official_bounds = [50, 100, 200, 300, 400]
cat_colors = {'Good':'#2ecc71','Satisfactory':'#f1c40f','Moderate':'#e67e22',
              'Poor':'#e74c3c','Very Poor':'#c0392b','Severe':'#8e44ad'}

fig, ax = plt.subplots(figsize=(14, 7))
sns.set_style('whitegrid')
for i, (start, end) in enumerate(zip([0]+official_bounds, official_bounds+[500])):
    ax.axvspan(start, end, alpha=0.08, color=list(cat_colors.values())[i], zorder=0)
ax.plot(bin_stats['aqi_bin'], bin_stats['mean'], color='#1f77b4', linewidth=2.5, label='Mean TRB')
ax.fill_between(bin_stats['aqi_bin'],
                bin_stats['mean'] - bin_stats['ci95'],
                bin_stats['mean'] + bin_stats['ci95'],
                color='#1f77b4', alpha=0.2, label='95% CI')
for ob in official_bounds:
    ax.axvline(ob, color='#555555', linestyle='--', linewidth=1.5, alpha=0.8)
for bp in breakpoints:
    ax.axvline(bp, color='#d62728', linestyle='-', linewidth=3, alpha=0.9)
ymid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
for bp in breakpoints:
    ax.text(bp, ymid, f'{bp:.0f}', rotation=90, va='center', ha='center',
            fontsize=16, color='#d62728', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
ax.set_xlabel('Air Quality Index (AQI)', fontsize=14)
ax.set_ylabel('Total Respiratory Burden (TRB) – µg/hour', fontsize=14)
ax.set_title('Official AQI Categories vs. Physiological Dose Breaks', fontsize=12, fontweight='bold')
ax.set_xlim(0, 500)
ax.set_ylim(bottom=0)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', alpha=0.3, label='Mean TRB ± 95% CI'),
                   plt.Line2D([0], [0], color='#555555', linestyle='--', lw=1.5, label='Official boundaries'),
                   plt.Line2D([0], [0], color='#d62728', lw=3, label='New breakpoints')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
plt.tight_layout()
fig.savefig(figures_dir / 'breakpoints_shock.png', dpi=300, bbox_inches='tight')
fig.savefig(figures_dir / 'breakpoints_shock.svg', dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved to {figures_dir}")