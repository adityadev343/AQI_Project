# -*- coding: utf-8 -*-
"""EDA - Portable version (no Google Drive / Colab specific paths)"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import ensure_dirs, DATASET_PATH, RESULTS_DIR, FIGURES_DIR, CITIES, RAW_POLLUTANTS, AQI_CATEGORIES, REPR_TARGET, PROFILES, SCENARIOS, AQI_COL, AQI_CAT_COL, SUBINDEX_COLS, REF_COLS, ALL_TARGETS

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

# Output folder inside results
OUT = RESULTS_DIR / "eda_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])
df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")

# Column groups (used across all sections)
RAW_COLS      = RAW_POLLUTANTS
AVG_24_COLS   = ['pm2.5_24hr', 'pm10_24hr', 'no2_24hr', 'so2_24hr', 'nh3_24hr']
AVG_8_COLS    = ['co_8hr', 'o3_8hr']
SUB_COLS      = SUBINDEX_COLS
AQI_COLS      = [AQI_COL, AQI_CAT_COL]
REF_WHO_COLS  = ['TRB_WHO_ref_m_s',  'TRB_WHO_ref_m_w', 'TRB_WHO_ref_f_s', 'TRB_WHO_ref_f_w']
REF_NAAQS_COLS= ['TRB_NAAQS_ref_m_s','TRB_NAAQS_ref_m_w','TRB_NAAQS_ref_f_s','TRB_NAAQS_ref_f_w']
REF_COLS      = REF_WHO_COLS + REF_NAAQS_COLS
PROFILES      = ['m_s', 'm_w', 'f_s', 'f_w']
SCENARIOS     = ['low', 'mid', 'high']
CITIES        = ['Delhi', 'Mumbai', 'Hyderabad']
AQI_CAT_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
CITY_COLORS   = {'Delhi': '#C0392B', 'Mumbai': '#2471A3', 'Hyderabad': '#1E8449'}

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

print('Dataset loaded successfully.')
print(f'Shape: {df.shape}')

# ============================================================================
# Section 3 — Structural Verification
# ============================================================================
print('='*65)
print('3.1  SHAPE')
print('='*65)
print(f'Rows    : {df.shape[0]:,}  (expected 56,880)')
print(f'Columns : {df.shape[1]}   (expected 69)')
print()
print('All 69 column names:')
for i, c in enumerate(df.columns, 1):
    print(f'  {i:>2}. {c}')

print()
print('='*65)
print('3.2  ROW COUNTS PER CITY')
print('='*65)
city_counts = df['city'].value_counts().sort_index()
print(city_counts.to_string())
print(f'\nExpected per city : ~18,960')

print()
print('='*65)
print('3.3  DUPLICATE (city, datetime) PAIRS')
print('='*65)
n_dups = df.duplicated(['city', 'datetime']).sum()
print(f'Duplicate rows found: {n_dups}')
if n_dups > 0:
    df = df.drop_duplicates(['city', 'datetime']).reset_index(drop=True)
    print(f'Duplicates dropped. New shape: {df.shape}')
else:
    print('No duplicates — dataset is clean.')

print()
print('='*65)
print('3.4  DATETIME RANGE AND MAX GAP PER CITY')
print('='*65)
for city in CITIES:
    cdf = df[df['city'] == city]['datetime'].sort_values()
    max_gap = cdf.diff().max()
    print(f'{city:<12}  Start: {cdf.min()}  |  End: {cdf.max()}  |  Max gap: {max_gap}')
print()
print('Expected range : 2024-01-01 00:00 → 2026-02-28 23:00')
print('Expected max gap : 0 days 01:00:00  (no missing hours)')

print()
print('='*65)
print('3.5  REFERENCE TRB CONSTANCY PER CITY')
print('='*65)
ref_std = df.groupby('city')[REF_COLS].std()
print(ref_std.to_string())
all_zero = (ref_std == 0).all().all()
print(f'\nAll reference TRB columns constant within each city: {all_zero}')

print()
print('='*65)
print('3.6  CO UNIT VERIFICATION')
print('='*65)
co_max = df['co'].max()
print(f'co column max : {co_max:.4f} mg/m³  (expected ≈ 9.80 mg/m³ from Cleaning Summary)')
if abs(co_max - 9.80) < 0.5:
    print('✓  CO is correctly in mg/m³')
else:
    print('✗  WARNING: CO max deviates significantly — check units')

trb_col = 'TRB_actual_m_s_mid'
print(f'TRB_actual_m_s_mid range check (should be µg/hour, typically hundreds to thousands):')
print(f'  Min : {df[trb_col].min():,.2f} µg/hour')
print(f'  Mean: {df[trb_col].mean():,.2f} µg/hour')
print(f'  Max : {df[trb_col].max():,.2f} µg/hour')

# ============================================================================
# Section 4 — Missing Value Profile
# ============================================================================
total_rows = len(df)

null_counts = df.isnull().sum()
null_pct    = (null_counts / total_rows * 100).round(4)

def col_group(col):
    if col in ['city', 'datetime']:         return '1. Identifiers'
    if col in RAW_COLS:                     return '2. Raw pollutants'
    if col in AVG_24_COLS:                  return '3. 24hr running averages'
    if col in AVG_8_COLS:                   return '4. 8hr running averages'
    if col in SUB_COLS:                     return '5. Sub-indices'
    if col in AQI_COLS:                     return '6. AQI columns'
    if col in REF_COLS:                     return '7. Reference TRB'
    if 'TRB_actual' in col:                 return '8. TRB actual'
    if col.startswith('TRBI'):              return '9. TRBI'
    return '10. Other'

null_df = pd.DataFrame({
    'Column'     : null_counts.index,
    'Group'      : [col_group(c) for c in null_counts.index],
    'NaN Count'  : null_counts.values,
    'NaN %'      : null_pct.values
})

group_summary = (
    null_df.groupby('Group')
    .agg(Columns=('Column','count'),
         Total_NaNs=('NaN Count','sum'),
         Max_NaN_pct=('NaN %','max'))
    .reset_index()
)

print('='*65)
print('4.1  NaN SUMMARY BY COLUMN GROUP')
print('='*65)
print(group_summary.to_string(index=False))

print()
print('='*65)
print('4.2  FULL COLUMN-LEVEL NaN TABLE')
print('='*65)
cols_with_nans = null_df[null_df['NaN Count'] > 0]
if len(cols_with_nans) == 0:
    print('No NaN values found in any column.')
else:
    print(cols_with_nans.to_string(index=False))

print()
print('='*65)
print('4.3  VALIDATION AGAINST EXPECTED NaN PATTERN')
print('='*65)

raw_nans = df[RAW_COLS].isnull().sum().sum()
print(f'Raw pollutant NaNs     : {raw_nans}  (expected 0)')

avg24_nans = df[AVG_24_COLS].isnull().sum().sum()
print(f'24hr average NaNs      : {avg24_nans}  (expected ~225)')

avg8_nans = df[AVG_8_COLS].isnull().sum().sum()
print(f'8hr average NaNs       : {avg8_nans}  (expected ~42)')

trb_actual_cols = [c for c in df.columns if 'TRB_actual' in c]
trbi_cols       = [c for c in df.columns if c.startswith('TRBI')]
trb_nans  = df[trb_actual_cols].isnull().sum().sum()
trbi_nans = df[trbi_cols].isnull().sum().sum()
print(f'TRB_actual NaNs        : {trb_nans}  (expected 0)')
print(f'TRBI NaNs              : {trbi_nans}  (expected 0)')

print()
print('='*65)
print('4.4  INVALID AQI ROWS PER CITY')
print('='*65)
for city in CITIES:
    cdf = df[df['city'] == city]
    invalid = cdf['aqi'].isna().sum()
    pct     = invalid / len(cdf) * 100
    flag    = '  ← INVESTIGATE (>5%)' if pct > 5 else ''
    print(f'{city:<12}  Invalid AQI rows: {invalid:>5}  ({pct:.2f}%){flag}')

null_df.to_csv(f'{OUT}/s4_nan_profile.csv', index=False)
print(f'\nSaved → {OUT}/s4_nan_profile.csv')

# ============================================================================
# Section 5 — Descriptive Statistics of Raw Pollutants Per City
# ============================================================================
stat_rows = []
for city in CITIES:
    cdf = df[df['city'] == city]
    for col in RAW_COLS:
        s = cdf[col].dropna()
        stat_rows.append({
            'City'    : city,
            'Pollutant': col,
            'Mean'    : round(s.mean(), 3),
            'Median'  : round(s.median(), 3),
            'Std'     : round(s.std(), 3),
            'IQR'     : round(s.quantile(0.75) - s.quantile(0.25), 3),
            'P95'     : round(s.quantile(0.95), 3),
            'P99'     : round(s.quantile(0.99), 3),
            'Max'     : round(s.max(), 3),
            'Skewness': round(s.skew(), 3)
        })

stats_df = pd.DataFrame(stat_rows)

print('='*65)
print('5.1  DESCRIPTIVE STATISTICS BY CITY AND POLLUTANT')
print('='*65)
for city in CITIES:
    print(f'\n── {city} ──')
    print(stats_df[stats_df['City'] == city].drop(columns='City').to_string(index=False))

print()
print('='*65)
print('5.2  CITY THAT DRIVES POOLED MAXIMUM PER POLLUTANT')
print('='*65)
for col in RAW_COLS:
    max_vals = {city: df[df['city']==city][col].max() for city in CITIES}
    driver   = max(max_vals, key=max_vals.get)
    print(f'{col:<12}  Max: {max_vals[driver]:>8.3f}  →  driven by {driver}')
    for c, v in max_vals.items():
        print(f'             {c}: {v:.3f}')

print()
print('='*65)
print('5.3  OUTLIER CHARACTERISATION (hours > 99th percentile per pollutant per city)')
print('='*65)
outlier_rows = []
for city in CITIES:
    cdf = df[df['city'] == city].copy()
    for col in RAW_COLS:
        p99    = cdf[col].quantile(0.99)
        outs   = cdf[cdf[col] > p99].copy()
        n_outs = len(outs)
        if n_outs > 0:
            month_counts = outs['datetime'].dt.month.value_counts().sort_index()
            top_months   = ', '.join([f'Month {m} ({cnt})' for m, cnt in month_counts.head(3).items()])
        else:
            top_months = 'None'
        outlier_rows.append({
            'City': city, 'Pollutant': col,
            'P99 threshold': round(p99, 3),
            'Hours > P99': n_outs,
            'Top months (hour count)': top_months
        })

outlier_df = pd.DataFrame(outlier_rows)
for city in CITIES:
    print(f'\n── {city} ──')
    print(outlier_df[outlier_df['City']==city].drop(columns='City').to_string(index=False))

stats_df.to_csv(f'{OUT}/s5_descriptive_stats.csv', index=False)
outlier_df.to_csv(f'{OUT}/s5_outlier_characterisation.csv', index=False)
print(f'\nSaved → {OUT}/s5_descriptive_stats.csv')
print(f'Saved → {OUT}/s5_outlier_characterisation.csv')

# ============================================================================
# Section 6 — AQI Category Distribution Per City
# ============================================================================
cat_rows = []
for city in CITIES:
    cdf       = df[df['city'] == city]
    valid_aqi = cdf.dropna(subset=['aqi_category'])
    counts    = valid_aqi['aqi_category'].value_counts()
    total_valid = len(valid_aqi)
    for cat in AQI_CAT_ORDER:
        cnt = counts.get(cat, 0)
        pct = cnt / len(cdf) * 100
        cat_rows.append({
            'City': city, 'Category': cat,
            'Count': cnt, 'Pct_of_city': round(pct, 2)
        })

cat_df = pd.DataFrame(cat_rows)

print('='*65)
print('6.1  AQI CATEGORY FREQUENCY TABLE (% of all rows incl. NaN)')
print('='*65)
pivot = cat_df.pivot(index='Category', columns='City', values='Count')[CITIES]
pivot_pct = cat_df.pivot(index='Category', columns='City', values='Pct_of_city')[CITIES]
pivot_display = pivot.copy().astype(str)
for city in CITIES:
    pivot_display[city] = pivot[city].astype(str) + '  (' + pivot_pct[city].astype(str) + '%)'
pivot_display = pivot_display.reindex(AQI_CAT_ORDER)
print(pivot_display.to_string())

print()
print('='*65)
print('6.2  CATEGORIES WITH < 100 VALID HOURS (low statistical power warning)')
print('='*65)
flagged = cat_df[cat_df['Count'] < 100]
if len(flagged) == 0:
    print('No categories below 100 hours in any city.')
else:
    print(flagged.to_string(index=False))
    print('\n→ Within-bucket regression will have low power for these.')

print()
print('='*65)
print('6.3  INVALID AQI FRACTION PER CITY')
print('='*65)
for city in CITIES:
    cdf   = df[df['city'] == city]
    n_nan = cdf['aqi'].isna().sum()
    pct   = n_nan / len(cdf) * 100
    flag  = '  ← INVESTIGATE' if pct > 5 else ''
    print(f'{city:<12}  NaN AQI: {n_nan:>5}  ({pct:.2f}%){flag}')

# Stacked bar chart
cat_colors = {
    'Good'               : '#2ECC71',
    'Satisfactory'       : '#82E0AA',
    'Moderate'           : '#F7DC6F',
    'Poor'               : '#F39C12',
    'Very Poor'          : '#E74C3C',
    'Severe'             : '#7B241C'
}

fig, ax = plt.subplots(figsize=(9, 5))
bottoms = np.zeros(len(CITIES))
x = np.arange(len(CITIES))

for cat in AQI_CAT_ORDER:
    vals = []
    for city in CITIES:
        row = cat_df[(cat_df['City']==city) & (cat_df['Category']==cat)]
        vals.append(row['Pct_of_city'].values[0] if len(row) > 0 else 0)
    ax.bar(x, vals, bottom=bottoms, label=cat,
           color=cat_colors[cat], edgecolor='white', linewidth=0.5)
    bottoms += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels(CITIES, fontsize=12)
ax.set_ylabel('% of total hours', fontsize=11)
ax.set_title('AQI Category Distribution by City', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, title='AQI Category')
ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(f'{OUT}/s6_aqi_category_distribution.png', dpi=150)
plt.show()
print(f'Saved → {OUT}/s6_aqi_category_distribution.png')

cat_df.to_csv(f'{OUT}/s6_aqi_category_counts.csv', index=False)
print(f'Saved → {OUT}/s6_aqi_category_counts.csv')

# ============================================================================
# Section 7 — Responsible Pollutant Distribution
# ============================================================================
valid_df = df.dropna(subset=['aqi']).copy()

def get_responsible(row):
    for col in SUB_COLS:
        if pd.notna(row[col]) and round(row[col], 2) == round(row['aqi'], 2):
            return col.replace('_subindex', '')
    vals = {col: row[col] for col in SUB_COLS if pd.notna(row[col])}
    if vals:
        return max(vals, key=vals.get).replace('_subindex', '')
    return None

valid_df['responsible_pollutant'] = valid_df.apply(get_responsible, axis=1)

POLL_LABELS = ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']
POLL_COLORS = ['#E74C3C','#E67E22','#3498DB','#F1C40F','#8E44AD','#2ECC71','#1ABC9C']

print('='*65)
print('7.1  RESPONSIBLE POLLUTANT FREQUENCY PER CITY')
print('='*65)

resp_rows = []
for city in CITIES:
    cdf   = valid_df[valid_df['city'] == city]
    counts = cdf['responsible_pollutant'].value_counts()
    total  = len(cdf)
    for poll in POLL_LABELS:
        cnt = counts.get(poll, 0)
        resp_rows.append({'City': city, 'Pollutant': poll,
                          'Count': cnt, 'Pct': round(cnt/total*100, 2)})

resp_df = pd.DataFrame(resp_rows)
for city in CITIES:
    print(f'\n── {city} ──')
    sub = resp_df[resp_df['City']==city].sort_values('Count', ascending=False)
    print(sub.drop(columns='City').to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, city in zip(axes, CITIES):
    sub = resp_df[resp_df['City']==city].sort_values('Count', ascending=False)
    ax.bar(sub['Pollutant'], sub['Pct'],
           color=[POLL_COLORS[POLL_LABELS.index(p)] for p in sub['Pollutant']],
           edgecolor='white')
    ax.set_title(city, fontweight='bold')
    ax.set_xlabel('Pollutant')
    ax.set_ylabel('% of valid AQI hours')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Responsible Pollutant Distribution by City', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/s7_responsible_pollutant.png', dpi=150)
plt.show()
print(f'\nSaved → {OUT}/s7_responsible_pollutant.png')

# Sub-index gap analysis
print()
print('='*65)
print('7.2  GAP BETWEEN HIGHEST AND SECOND-HIGHEST SUB-INDEX (overall)')
print('='*65)

def subindex_gap(row):
    vals = sorted([row[c] for c in SUB_COLS if pd.notna(row[c])], reverse=True)
    if len(vals) >= 2:
        return vals[0] - vals[1]
    return np.nan

valid_df['subindex_gap'] = valid_df.apply(subindex_gap, axis=1)

gap_rows = []
for city in CITIES:
    g = valid_df[valid_df['city']==city]['subindex_gap'].dropna()
    gap_rows.append({
        'City'       : city,
        'Mean gap'   : round(g.mean(), 3),
        'Median gap' : round(g.median(), 3),
        'Fraction gap < 20': round((g < 20).sum() / len(g) * 100, 2)
    })

gap_summary = pd.DataFrame(gap_rows)
print(gap_summary.to_string(index=False))
print('\n→ "Fraction gap < 20" = % of hours where a second pollutant is close to driving AQI.')

# Seasonal variation of gap
print()
print('='*65)
print('7.3  SEASONAL PATTERN OF SUB-INDEX GAP')
print('='*65)

if 'month' not in valid_df.columns:
    valid_df['month'] = valid_df['datetime'].dt.month

gap_seasonal = (
    valid_df.groupby(['city', 'month'])['subindex_gap']
    .agg(['mean', 'median'])
    .reset_index()
    .rename(columns={'mean': 'Mean gap', 'median': 'Median gap'})
)

gap_lt20 = (
    valid_df[valid_df['subindex_gap'] < 20]
    .groupby(['city', 'month'])
    .size()
    .reset_index(name='count_lt20')
)
total_per_month = valid_df.groupby(['city', 'month']).size().reset_index(name='total')
gap_lt20 = gap_lt20.merge(total_per_month, on=['city', 'month'])
gap_lt20['Frac_gap_lt20_pct'] = (gap_lt20['count_lt20'] / gap_lt20['total']) * 100

gap_seasonal = gap_seasonal.merge(gap_lt20[['city', 'month', 'Frac_gap_lt20_pct']],
                                  on=['city', 'month'], how='left').fillna(0)

gap_seasonal['Mean gap'] = gap_seasonal['Mean gap'].round(2)
gap_seasonal['Median gap'] = gap_seasonal['Median gap'].round(2)
gap_seasonal['Frac_gap_lt20_pct'] = gap_seasonal['Frac_gap_lt20_pct'].round(2)

for city in CITIES:
    print(f'\n── {city} ──')
    sub = gap_seasonal[gap_seasonal['city']==city].sort_values('month')
    print(sub.drop(columns='city').to_string(index=False))

gap_seasonal.to_csv(f'{OUT}/s7_subindex_gap_seasonal.csv', index=False)

fig, ax = plt.subplots(figsize=(11, 5))
for city in CITIES:
    sub = gap_seasonal[gap_seasonal['city']==city].sort_values('month')
    ax.plot(sub['month'], sub['Mean gap'], marker='o', label=city,
            color=CITY_COLORS[city], linewidth=2, markersize=6)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_NAMES[m] for m in range(1, 13)])
ax.set_xlabel('Month')
ax.set_ylabel('Mean gap (highest - 2nd highest sub-index)')
ax.set_title('Seasonal Pattern of Sub‑Index Gap by City', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/s7_subindex_gap_seasonal.png', dpi=150)
plt.show()
print(f'Saved → {OUT}/s7_subindex_gap_seasonal.csv')
print(f'Saved → {OUT}/s7_subindex_gap_seasonal.png')

resp_df.to_csv(f'{OUT}/s7_responsible_pollutant.csv', index=False)
gap_summary.to_csv(f'{OUT}/s7_subindex_gap.csv', index=False)
print(f'\nSaved → {OUT}/s7_responsible_pollutant.csv')
print(f'Saved → {OUT}/s7_subindex_gap.csv')

# ============================================================================
# Section 8 — Pollutant Correlations Per City
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
corr_matrices = {}

for ax, city in zip(axes, CITIES):
    cdf  = df[df['city'] == city][RAW_COLS].dropna()
    corr = cdf.corr(method='spearman')
    corr_matrices[city] = corr
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        linewidths=0.5, cbar_kws={'shrink': 0.7},
        annot_kws={'size': 8}
    )
    ax.set_title(f'Spearman Correlation — {city}', fontweight='bold', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.suptitle('Spearman Correlation of Raw Pollutants by City (lower triangle)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/s8_spearman_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved → {OUT}/s8_spearman_heatmaps.png')

print()
print('='*65)
print('8.1  PAIRS WITH |r| > 0.70 IN AT LEAST 2 CITIES')
print('='*65)

from itertools import combinations
high_corr = {}
for p1, p2 in combinations(RAW_COLS, 2):
    cities_above = []
    for city in CITIES:
        r = corr_matrices[city].loc[p1, p2]
        if abs(r) > 0.70:
            cities_above.append(f'{city} (r={r:.2f})')
    if len(cities_above) >= 2:
        high_corr[f'{p1} vs {p2}'] = cities_above

if high_corr:
    for pair, cities_list in high_corr.items():
        print(f'  {pair:<25}  {"  |  ".join(cities_list)}')
else:
    print('  No pollutant pair exceeds |r| > 0.70 in 2 or more cities.')

print()
print('='*65)
print('8.2  O₃ vs NO₂ CORRELATION (expected negative due to titration chemistry)')
print('='*65)
for city in CITIES:
    r = corr_matrices[city].loc['o3', 'no2']
    direction = 'negative ✓' if r < 0 else 'positive — unexpected'
    print(f'  {city:<12}  r(O₃, NO₂) = {r:+.3f}  →  {direction}')

for city in CITIES:
    corr_matrices[city].to_csv(f'{OUT}/s8_spearman_{city.lower()}.csv')
print(f'\nSaved → {OUT}/s8_spearman_[city].csv (3 files)')

# ============================================================================
# Section 9 — Temporal Patterns
# ============================================================================
df['month']    = df['datetime'].dt.month
df['hour']     = df['datetime'].dt.hour
df['date']     = df['datetime'].dt.date
df['year_month'] = df['datetime'].dt.to_period('M')

print('='*65)
print('9.1  SEASONAL PATTERN — MEAN AQI PER MONTH')
print('='*65)

seasonal = (
    df.dropna(subset=['aqi'])
    .groupby(['city', 'month'])['aqi']
    .mean().round(2)
    .reset_index()
)

print('Peak-to-trough ratio (worst month mean AQI / best month mean AQI):')
for city in CITIES:
    sub   = seasonal[seasonal['city']==city]
    worst = sub.loc[sub['aqi'].idxmax()]
    best  = sub.loc[sub['aqi'].idxmin()]
    ratio = worst['aqi'] / best['aqi']
    print(f'  {city:<12}  Worst: {MONTH_NAMES[int(worst["month"])]} ({worst["aqi"]:.1f})  '
          f'Best: {MONTH_NAMES[int(best["month"])]} ({best["aqi"]:.1f})  Ratio: {ratio:.2f}x')

fig, ax = plt.subplots(figsize=(11, 5))
for city in CITIES:
    sub = seasonal[seasonal['city']==city].sort_values('month')
    ax.plot(sub['month'], sub['aqi'], marker='o', label=city,
            color=CITY_COLORS[city], linewidth=2, markersize=4)
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_NAMES[m] for m in range(1, 13)])
ax.set_xlabel('Month')
ax.set_ylabel('Mean AQI')
ax.set_title('Seasonal Pattern — Mean AQI by Month and City', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
for boundary, label in [(50,'Good'), (100,'Satisfactory'), (200,'Moderate'),
                         (300,'Poor'), (400,'Very Poor')]:
    ax.axhline(boundary, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()
fig.savefig(f'{OUT}/s9_seasonal_aqi.png', dpi=150)
plt.show()
print(f'\nSaved → {OUT}/s9_seasonal_aqi.png')

print()
print('='*65)
print('9.2  DIURNAL PATTERN — MEAN AQI PER HOUR OF DAY')
print('='*65)

diurnal = (
    df.dropna(subset=['aqi'])
    .groupby(['city', 'hour'])['aqi']
    .mean().round(2)
    .reset_index()
)

print('Peak-hour vs trough-hour difference:')
for city in CITIES:
    sub   = diurnal[diurnal['city']==city]
    peak  = sub.loc[sub['aqi'].idxmax()]
    trough= sub.loc[sub['aqi'].idxmin()]
    diff  = peak['aqi'] - trough['aqi']
    flag  = ' ← meaningful feature candidate' if diff > 30 else ''
    print(f'  {city:<12}  Peak: {int(peak["hour"]):02d}:00 ({peak["aqi"]:.1f})  '
          f'Trough: {int(trough["hour"]):02d}:00 ({trough["aqi"]:.1f})  '
          f'Diff: {diff:.1f}{flag}')

fig, ax = plt.subplots(figsize=(11, 5))
for city in CITIES:
    sub = diurnal[diurnal['city']==city].sort_values('hour')
    ax.plot(sub['hour'], sub['aqi'], marker='o', label=city,
            color=CITY_COLORS[city], linewidth=2, markersize=4)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], rotation=45)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Mean AQI')
ax.set_title('Diurnal Pattern — Mean AQI by Hour of Day and City', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/s9_diurnal_aqi.png', dpi=150)
plt.show()
print(f'\nSaved → {OUT}/s9_diurnal_aqi.png')

print()
print('='*65)
print('9.3  DATA AVAILABILITY CALENDAR (fraction of 24 hours with valid AQI per day)')
print('='*65)

avail = (
    df.groupby(['city', 'date'])
    .apply(lambda x: x['aqi'].notna().sum() / 24 * 100)
    .reset_index()
    .rename(columns={0: 'valid_pct'})
)
avail['date'] = pd.to_datetime(avail['date'])
avail['year_month'] = avail['date'].dt.to_period('M').astype(str)
avail['day']        = avail['date'].dt.day

fig, axes = plt.subplots(3, 1, figsize=(18, 12))
for ax, city in zip(axes, CITIES):
    sub = avail[avail['city']==city]
    pivot_cal = sub.pivot(index='year_month', columns='day', values='valid_pct')
    pivot_cal = pivot_cal.sort_index()
    sns.heatmap(
        pivot_cal, ax=ax, cmap='RdYlGn', vmin=0, vmax=100,
        linewidths=0.3, linecolor='white',
        cbar_kws={'label': '% hours with valid AQI', 'shrink': 0.6}
    )
    ax.set_title(f'{city} — Daily AQI Availability', fontweight='bold')
    ax.set_xlabel('Day of Month')
    ax.set_ylabel('Month')
    ax.tick_params(axis='y', labelsize=8)

    low_avail = sub[sub['valid_pct'] < 80]
    if len(low_avail) > 0:
        print(f'  {city}: {len(low_avail)} days with < 80% valid AQI hours')
    else:
        print(f'  {city}: All days have ≥ 80% valid AQI hours ✓')

plt.suptitle('Data Availability Calendar — % of Daily Hours with Valid AQI',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(f'{OUT}/s9_availability_calendar.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved → {OUT}/s9_availability_calendar.png')

seasonal.to_csv(f'{OUT}/s9_seasonal_aqi.csv', index=False)
diurnal.to_csv(f'{OUT}/s9_diurnal_aqi.csv', index=False)
print(f'Saved → {OUT}/s9_seasonal_aqi.csv')
print(f'Saved → {OUT}/s9_diurnal_aqi.csv')

# ============================================================================
# Section 10 — TRB and TRBI Overview
# ============================================================================
print('='*65)
print('10.1  TRB UNCERTAINTY BAND')
print('     (TRB_actual_high - TRB_actual_low) / TRB_actual_mid × 100')
print('='*65)

unc_rows = []
for city in CITIES:
    cdf = df[df['city'] == city]
    for prof in PROFILES:
        low  = cdf[f'TRB_actual_{prof}_low']
        mid  = cdf[f'TRB_actual_{prof}_mid']
        high = cdf[f'TRB_actual_{prof}_high']
        band = ((high - low) / mid * 100).replace([np.inf, -np.inf], np.nan).dropna()
        unc_rows.append({
            'City'                  : city,
            'Profile'               : prof,
            'Mean uncertainty (%)'  : round(band.mean(), 3),
            'P95 uncertainty (%)'   : round(band.quantile(0.95), 3)
        })

unc_df = pd.DataFrame(unc_rows)
for city in CITIES:
    print(f'\n── {city} ──')
    print(unc_df[unc_df['City']==city].drop(columns='City').to_string(index=False))

print('\n→ If mean uncertainty < 10%: low/mid/high scenarios are nearly equivalent.')
print('  If mean uncertainty > 25%: scenario choice substantially affects conclusions.')

print()
print('='*65)
print('10.2  FRACTION OF HOURS WHERE TRBI > 1 (mid scenario only)')
print('     WHO: PM2.5 limit = 15 µg/m³  |  NAAQS: PM2.5 limit = 60 µg/m³')
print('     Expected: TRBI_WHO > 1 fraction >> TRBI_NAAQS > 1 fraction')
print('='*65)

trbi_rows = []
for city in CITIES:
    cdf = df[df['city'] == city]
    for prof in PROFILES:
        who_col   = f'TRBI_WHO_{prof}_mid'
        naaqs_col = f'TRBI_NAAQS_{prof}_mid'
        who_valid = cdf[who_col].notna()
        naaqs_valid = cdf[naaqs_col].notna()
        who_frac  = round((cdf.loc[who_valid, who_col] > 1).sum() / who_valid.sum() * 100, 2) if who_valid.any() else np.nan
        naaqs_frac= round((cdf.loc[naaqs_valid, naaqs_col] > 1).sum() / naaqs_valid.sum() * 100, 2) if naaqs_valid.any() else np.nan
        trbi_rows.append({
            'City'              : city,
            'Profile'           : prof,
            'TRBI_WHO > 1 (%)'  : who_frac,
            'TRBI_NAAQS > 1 (%)': naaqs_frac
        })

trbi_df = pd.DataFrame(trbi_rows)
for city in CITIES:
    print(f'\n── {city} ──')
    print(trbi_df[trbi_df['City']==city].drop(columns='City').to_string(index=False))

print()
print('VERIFICATION: Is TRBI_WHO > 1 fraction always > TRBI_NAAQS > 1 fraction?')
check = (trbi_df['TRBI_WHO > 1 (%)'] > trbi_df['TRBI_NAAQS > 1 (%)']).all()
if check:
    print('  ✓  Yes — reference computation is consistent with stricter WHO limits.')
else:
    print('  ✗  No — some rows violate expectation. Review TRB_ref calculation.')
    print(trbi_df[trbi_df['TRBI_WHO > 1 (%)'] <= trbi_df['TRBI_NAAQS > 1 (%)']].to_string())

unc_df.to_csv(f'{OUT}/s10_trb_uncertainty_band.csv', index=False)
trbi_df.to_csv(f'{OUT}/s10_trbi_gt1_fraction.csv', index=False)
print(f'\nSaved → {OUT}/s10_trb_uncertainty_band.csv')
print(f'Saved → {OUT}/s10_trbi_gt1_fraction.csv')

# ============================================================================
# Section 11 — Summary Findings Table
# ============================================================================
summary_rows = []

for city in CITIES:
    cdf       = df[df['city'] == city]
    valid_aqi = cdf.dropna(subset=['aqi'])
    n_total   = len(cdf)
    n_valid   = len(valid_aqi)

    cat_counts = valid_aqi['aqi_category'].value_counts()
    most_freq_cat  = cat_counts.idxmax() if len(cat_counts) > 0 else 'N/A'
    rarest_cat     = cat_counts.idxmin() if len(cat_counts) > 0 else 'N/A'
    rarest_count   = int(cat_counts.min()) if len(cat_counts) > 0 else 0

    resp_city  = resp_df[resp_df['City']==city].sort_values('Count', ascending=False)
    top_resp   = resp_city.iloc[0]['Pollutant'] if len(resp_city) > 0 else 'N/A'

    gap_city   = gap_summary[gap_summary['City']==city]
    mean_gap   = gap_city['Mean gap'].values[0] if len(gap_city) > 0 else np.nan

    seas_city  = seasonal[seasonal['city']==city]
    if len(seas_city) > 0:
        peak_row   = seas_city.loc[seas_city['aqi'].idxmax()]
        peak_month = f"{MONTH_NAMES[int(peak_row['month'])]} ({peak_row['aqi']:.1f})"
    else:
        peak_month = 'N/A'

    trbi_city = trbi_df[(trbi_df['City']==city) & (trbi_df['Profile']=='m_s')]
    who_frac  = trbi_city['TRBI_WHO > 1 (%)'].values[0]   if len(trbi_city) > 0 else np.nan
    naaqs_frac= trbi_city['TRBI_NAAQS > 1 (%)'].values[0] if len(trbi_city) > 0 else np.nan

    unc_city  = unc_df[(unc_df['City']==city) & (unc_df['Profile']=='m_s')]
    unc_mean  = unc_city['Mean uncertainty (%)'].values[0] if len(unc_city) > 0 else np.nan

    summary_rows.append({
        'Finding'                                    : 'Value',
        'City'                                       : city,
        'Total valid rows'                           : n_total,
        'Valid AQI rows (%)'                         : f'{n_valid} ({n_valid/n_total*100:.2f}%)',
        'Most frequent AQI category'                 : most_freq_cat,
        'Rarest AQI category (count)'                : f'{rarest_cat} ({rarest_count})',
        'Responsible pollutant (most frequent)'      : top_resp,
        'Mean gap (highest vs 2nd highest sub-index)': round(mean_gap, 2),
        'Peak pollution month (mean AQI)'            : peak_month,
        'Fraction of hours TRBI_WHO_m_s_mid > 1 (%)'  : who_frac,
        'Fraction of hours TRBI_NAAQS_m_s_mid > 1 (%)': naaqs_frac,
        'TRB uncertainty band mean % (m_s, mid)'    : unc_mean
    })

summary_df = pd.DataFrame(summary_rows).drop(columns='Finding')
summary_T  = summary_df.set_index('City').T

print('='*65)
print('11.  EDA SUMMARY FINDINGS TABLE')
print('='*65)
print(summary_T.to_string())

summary_T.to_csv(f'{OUT}/s11_summary_findings.csv')
print(f'\nSaved → {OUT}/s11_summary_findings.csv')

print()
print('='*65)
print('ALL EDA OUTPUT FILES')
print('='*65)
for f_name in sorted(os.listdir(OUT)):
    fp = OUT / f_name
    size_kb = fp.stat().st_size / 1024
    print(f'  {f_name:<50}  {size_kb:>6.1f} KB')

print('\nEDA complete.')