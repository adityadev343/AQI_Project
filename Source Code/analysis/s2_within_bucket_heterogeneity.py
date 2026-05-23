# -*- coding: utf-8 -*-
"""Session 2: Within-Bucket Heterogeneity – Portable version"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LinearGAM, s as gam_s
from sklearn.metrics import r2_score
import pingouin as pg

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import (ensure_dirs, DATASET_PATH, RESULTS_DIR, FIGURES_DIR,
                         CITIES, AQI_CATEGORIES, ALL_TARGETS, TIER1_TARGETS,
                         REPR_TARGET, SUBINDEX_COLS, RANDOM_SEED)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

session_dir = RESULTS_DIR / "session_2"
session_dir.mkdir(parents=True, exist_ok=True)
fig_dir = FIGURES_DIR / "session2"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
print(f"Shape: {df.shape}")

# ----------------------------------------------------------------------
# Helper for sub-index gap
def subindex_gap(row):
    vals = [row[c] for c in SUBINDEX_COLS if pd.notna(row[c])]
    if len(vals) < 2:
        return 0.0
    sv = sorted(vals, reverse=True)
    return sv[0] - sv[1]

# ----------------------------------------------------------------------
# 1. Linear regression TRB ~ AQI within each bucket
print("\n=== 1. Linear regression within AQI buckets ===")
MIN_ROWS = 100
lin_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'])
    for cat in AQI_CATEGORIES:
        if cat == 'Severe' and city != 'Delhi':
            continue
        sub = dfc[dfc['aqi_category'] == cat]
        for target in ALL_TARGETS:
            sub_t = sub.dropna(subset=[target])
            if len(sub_t) < MIN_ROWS:
                continue
            X = sub_t['aqi'].values
            y = sub_t[target].values
            Xc = sm.add_constant(X)
            model = sm.OLS(y, Xc).fit()
            lin_results.append({
                'city': city, 'category': cat, 'target': target,
                'n': len(sub_t), 'slope': model.params[1], 'p_value': model.pvalues[1],
                'R2': model.rsquared, 'AIC': model.aic,
                'H2_supported': model.params[1] > 0 and model.pvalues[1] < 0.05
            })
df_lin = pd.DataFrame(lin_results)
df_lin.to_csv(session_dir / 'within_bucket_ols.csv', index=False)
print("Linear regression done.")

# ----------------------------------------------------------------------
# 2. GAM fitting + AIC comparison
print("\n=== 2. GAM vs linear (AIC, F-test) ===")
def approximate_f_test(y, y_pred_lin, y_pred_gam, n_params_lin=2, n_params_gam=5):
    n = len(y)
    rss_lin = np.sum((y - y_pred_lin)**2)
    rss_gam = np.sum((y - y_pred_gam)**2)
    df_diff = n_params_gam - n_params_lin
    df_res = n - n_params_gam
    if df_diff <= 0 or df_res <= 0 or rss_gam <= 0:
        return np.nan, np.nan
    F = ((rss_lin - rss_gam) / df_diff) / (rss_gam / df_res)
    p = 1 - stats.f.cdf(F, df_diff, df_res)
    return F, p

gam_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'])
    for cat in AQI_CATEGORIES:
        if cat == 'Severe' and city != 'Delhi':
            continue
        sub = dfc[dfc['aqi_category'] == cat]
        for target in ALL_TARGETS:
            sub_t = sub.dropna(subset=[target])
            if len(sub_t) < 50:
                continue
            x = sub_t['aqi'].values.reshape(-1,1)
            y = sub_t[target].values
            # Linear predictions
            lin_entry = df_lin[(df_lin['city']==city) & (df_lin['category']==cat) & (df_lin['target']==target)]
            if len(lin_entry) == 0:
                continue
            slope = lin_entry['slope'].values[0]
            intercept = lin_entry['intercept'].values[0] if 'intercept' in lin_entry else sub_t[target].mean()
            y_pred_lin = intercept + slope * x.ravel()
            aic_lin = lin_entry['AIC'].values[0]
            # GAM
            gam = LinearGAM(gam_s(0)).fit(x, y)
            y_pred_gam = gam.predict(x)
            rss_gam = np.sum((y - y_pred_gam)**2)
            edof = gam.statistics_['edof']
            aic_gam = len(y) * np.log(rss_gam/len(y) + 1e-10) + 2 * edof
            F_stat, p_ftest = approximate_f_test(y, y_pred_lin, y_pred_gam,
                                                 n_params_gam=max(3, int(edof)))
            gam_results.append({
                'city': city, 'category': cat, 'target': target,
                'n': len(sub_t), 'R2_gam': r2_score(y, y_pred_gam),
                'AIC_linear': aic_lin, 'AIC_gam': aic_gam,
                'delta_AIC': aic_gam - aic_lin,
                'F_stat': F_stat, 'p_ftest': p_ftest,
                'gam_preferred': aic_gam < aic_lin
            })
df_gam = pd.DataFrame(gam_results)
df_gam.to_csv(session_dir / 'gam_vs_linear.csv', index=False)
print("GAM fitting done.")

# ----------------------------------------------------------------------
# 3. ICC(3,1) and CV
print("\n=== 3. ICC(3,1) and coefficient of variation ===")
N_BINS = 10
icc_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'])
    for cat in AQI_CATEGORIES:
        if cat == 'Severe' and city != 'Delhi':
            continue
        sub = dfc[dfc['aqi_category'] == cat].copy()
        if len(sub) < 30:
            continue
        try:
            sub['aqi_bin'] = pd.qcut(sub['aqi'], q=N_BINS, labels=False, duplicates='drop')
        except:
            sub['aqi_bin'] = pd.cut(sub['aqi'], bins=N_BINS, labels=False)
        for target in ALL_TARGETS:
            sub_t = sub.dropna(subset=[target, 'aqi_bin'])
            if len(sub_t) < 30 or sub_t['aqi_bin'].nunique() < 2:
                continue
            y = sub_t[target].values
            cv = y.std() / y.mean() * 100 if y.mean() > 0 else np.nan
            # ICC
            icc_df = sub_t[['aqi_bin', target]].reset_index(drop=True)
            icc_df.columns = ['raters', 'ratings']
            icc_df['targets'] = icc_df.index
            try:
                icc_res = pg.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                icc_val = icc_res.set_index('Type').loc['ICC3','ICC']
                icc_ci = icc_res.set_index('Type').loc['ICC3','CI95%']
            except:
                icc_val = np.nan
                icc_ci = [np.nan, np.nan]
            icc_results.append({
                'city': city, 'category': cat, 'target': target,
                'n': len(sub_t), 'CV_pct': cv,
                'ICC3_1': icc_val, 'ICC_CI_low': icc_ci[0], 'ICC_CI_high': icc_ci[1]
            })
df_icc = pd.DataFrame(icc_results)
df_icc.to_csv(session_dir / 'icc_cv.csv', index=False)
print("ICC computed.")

# ----------------------------------------------------------------------
# 4. Mixed-effects model (TRB ~ AQI + (1|profile))
print("\n=== 4. Mixed-effects model (profile as random intercept) ===")
me_results = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'])
    # Long format for profiles
    profiles = ['m_s', 'm_w', 'f_s', 'f_w']
    long_list = []
    for prof in profiles:
        col = f'TRB_actual_{prof}_mid'
        if col in dfc.columns:
            tmp = dfc[['aqi', 'aqi_category', col]].copy()
            tmp = tmp.dropna()
            tmp['profile'] = prof
            tmp = tmp.rename(columns={col: 'TRB'})
            long_list.append(tmp)
    if not long_list:
        continue
    df_long = pd.concat(long_list)
    for cat in ['Moderate', 'Poor', 'Very Poor']:
        sub = df_long[df_long['aqi_category'] == cat]
        if len(sub) < 100:
            continue
        try:
            model = smf.mixedlm('TRB ~ aqi', sub, groups=sub['profile']).fit(reml=False)
            me_results.append({
                'city': city, 'category': cat,
                'slope_aqi': model.params.get('aqi', np.nan),
                'p_aqi': model.pvalues.get('aqi', np.nan),
                'AIC': model.aic
            })
        except Exception as e:
            print(f"Mixed model failed for {city} {cat}: {e}")
df_me = pd.DataFrame(me_results)
df_me.to_csv(session_dir / 'mixed_effects_slope.csv', index=False)
print("Mixed-effects done.")

# ----------------------------------------------------------------------
# 5. Seasonal decomposition (all cities)
print("\n=== 5. Seasonal within-bucket slopes ===")
SEASONS = {
    'Winter': [12,1,2], 'Summer': [3,4,5,6],
    'Monsoon': [7,8,9], 'Pre-winter': [10,11]
}
seasonal_rows = []
df['month'] = df['datetime'].dt.month
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'])
    for season, months in SEASONS.items():
        seas_df = dfc[dfc['month'].isin(months)]
        for cat in AQI_CATEGORIES:
            if cat == 'Severe' and city != 'Delhi':
                continue
            sub = seas_df[seas_df['aqi_category'] == cat]
            for target in ALL_TARGETS:
                sub_t = sub.dropna(subset=[target])
                if len(sub_t) < 20:
                    continue
                X = sub_t['aqi'].values
                y = sub_t[target].values
                Xc = sm.add_constant(X)
                model = sm.OLS(y, Xc).fit()
                seasonal_rows.append({
                    'city': city, 'season': season, 'category': cat, 'target': target,
                    'n': len(sub_t), 'slope': model.params[1], 'p_value': model.pvalues[1],
                    'R2': model.rsquared, 'H2_supported': model.params[1] > 0 and model.pvalues[1] < 0.05
                })
df_seasonal = pd.DataFrame(seasonal_rows)
df_seasonal.to_csv(session_dir / 'seasonal_slopes.csv', index=False)

# Plot seasonal slopes for REPR_TARGET
fig, axes = plt.subplots(1, 3, figsize=(18,7), sharey=True)
season_order = ['Winter', 'Summer', 'Monsoon', 'Pre-winter']
for ax, city in zip(axes, CITIES):
    sub = df_seasonal[(df_seasonal['target']==REPR_TARGET) & (df_seasonal['city']==city)]
    for cat in AQI_CATEGORIES:
        sub_cat = sub[sub['category']==cat]
        if sub_cat.empty:
            continue
        sub_cat = sub_cat.set_index('season').reindex(season_order).reset_index()
        ax.plot(sub_cat['season'], sub_cat['slope'], marker='o', label=cat)
    ax.axhline(0, color='black', ls='--')
    ax.set_title(city)
    ax.set_xlabel('Season')
    if ax == axes[0]:
        ax.set_ylabel('Slope (µg/hr per AQI unit)')
    ax.tick_params(axis='x', rotation=45)
plt.suptitle(f'Within-Bucket Slope by Season ({REPR_TARGET})')
plt.tight_layout()
fig.savefig(fig_dir / 'seasonal_slope_all_cities.png', dpi=300)
fig.savefig(fig_dir / 'seasonal_slope_all_cities.svg', dpi=300)
plt.close()

# ----------------------------------------------------------------------
# 6. Gap-stratified within-bucket slopes
print("\n=== 6. Stratified by sub-index gap (<20 vs >=20) ===")
GAP_THRESH = 20
gap_strat_rows = []
for city in CITIES:
    dfc = df[df['city'] == city].dropna(subset=['aqi', 'aqi_category'] + SUBINDEX_COLS).copy()
    dfc['gap'] = dfc.apply(subindex_gap, axis=1)
    for stratum, mask in [('low_gap', dfc['gap'] < GAP_THRESH), ('high_gap', dfc['gap'] >= GAP_THRESH)]:
        stratum_df = dfc[mask]
        if len(stratum_df) < 50:
            continue
        for cat in AQI_CATEGORIES:
            if cat == 'Severe' and city != 'Delhi':
                continue
            sub = stratum_df[stratum_df['aqi_category'] == cat]
            for target in TIER1_TARGETS:
                sub_t = sub.dropna(subset=[target])
                if len(sub_t) < 20:
                    continue
                X = sub_t['aqi'].values
                y = sub_t[target].values
                model = sm.OLS(y, sm.add_constant(X)).fit()
                gap_strat_rows.append({
                    'city': city, 'stratum': stratum, 'category': cat, 'target': target,
                    'n': len(sub_t), 'slope': model.params[1], 'p_value': model.pvalues[1], 'R2': model.rsquared
                })
df_gap_strat = pd.DataFrame(gap_strat_rows)
df_gap_strat.to_csv(session_dir / 'gap_stratified_slopes.csv', index=False)

# Plot comparison low vs high gap
fig, axes = plt.subplots(1,3,figsize=(18,5))
for ax, city in zip(axes, CITIES):
    sub = df_gap_strat[(df_gap_strat['target']==REPR_TARGET) & (df_gap_strat['city']==city)]
    if sub.empty:
        ax.set_title(city); continue
    cats = [c for c in AQI_CATEGORIES if c in sub['category'].values]
    x = np.arange(len(cats))
    for i, (stratum, color) in enumerate([('low_gap','steelblue'), ('high_gap','coral')]):
        vals = [sub[(sub['category']==c)&(sub['stratum']==stratum)]['slope'].values[0] if len(sub[(sub['category']==c)&(sub['stratum']==stratum)])>0 else np.nan for c in cats]
        ax.bar(x + i*0.35, vals, width=0.35, label=stratum, color=color, alpha=0.8)
    ax.axhline(0, color='black', ls='--')
    ax.set_xticks(x+0.175)
    ax.set_xticklabels(cats, rotation=30, ha='right')
    ax.set_title(city)
    if ax==axes[0]:
        ax.set_ylabel('Slope (µg/hr/AQI)')
        ax.legend()
plt.suptitle(f'Within-Bucket Slope: Low-Gap vs High-Gap ({REPR_TARGET})')
plt.tight_layout()
fig.savefig(fig_dir / 'gap_stratified_slopes.png', dpi=300)
fig.savefig(fig_dir / 'gap_stratified_slopes.svg', dpi=300)
plt.close()

print("\n=== Session 2 complete ===")
print(f"Results saved to {session_dir}")
print(f"Figures saved to {fig_dir}")