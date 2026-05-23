import json
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH) as f:
    _config = json.load(f)

BASE_DIR = Path(_config["base_dir"]).resolve()
RAW_DATA_DIR = BASE_DIR / _config["raw_data_dir"]
PROCESSED_DATA_DIR = BASE_DIR / _config["processed_data_dir"]
RESULTS_DIR = BASE_DIR / _config["results_dir"]
FIGURES_DIR = BASE_DIR / _config["figures_dir"]
MODELS_DIR = BASE_DIR / _config["models_dir"]
CHECKPOINTS_DIR = BASE_DIR / _config["checkpoints_dir"]

# Dataset paths
RAW_DATASET_PATH = RAW_DATA_DIR / _config["raw_data_filename"]
DATASET_PATH = PROCESSED_DATA_DIR / _config["dataset_filename"]

# Create all directories if they don't exist
def ensure_dirs():
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for i in range(1, 8):
        (RESULTS_DIR / f"session_{i}").mkdir(parents=True, exist_ok=True)

# Column definitions (same as before but without hardcoded paths)
CITIES = ['Delhi', 'Mumbai', 'Hyderabad']
CITY_COL = 'city'
DATETIME_COL = 'datetime'
AQI_COL = 'aqi'
AQI_CAT_COL = 'aqi_category'

RAW_POLLUTANTS = ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3']
ROLLING_24HR_COLS = ['pm2.5_24hr','pm10_24hr','no2_24hr','so2_24hr','nh3_24hr']
ROLLING_8HR_COLS = ['co_8hr', 'o3_8hr']
ROLLING_AVG_COLS = ROLLING_24HR_COLS + ROLLING_8HR_COLS
SUBINDEX_COLS = ['pm2.5_subindex','pm10_subindex','no2_subindex',
                 'so2_subindex','co_subindex','o3_subindex','nh3_subindex']

AQI_CATEGORIES = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']
AQI_BOUNDS = {
    'Good': (0, 50),
    'Satisfactory': (51, 100),
    'Moderate': (101, 200),
    'Poor': (201, 300),
    'Very Poor': (301, 400),
    'Severe': (401, float('inf')),
}

PROFILES = ['m_s', 'm_w', 'f_s', 'f_w']
SCENARIOS = ['low', 'mid', 'high']

TRB_ACTUAL_COLS = [f'TRB_actual_{p}_{s}' for p in PROFILES for s in SCENARIOS]
TRB_WHO_REF_COLS = [f'TRB_WHO_ref_{p}' for p in PROFILES]
TRB_NAAQS_REF_COLS = [f'TRB_NAAQS_ref_{p}' for p in PROFILES]
TRBI_WHO_COLS = [f'TRBI_WHO_{p}_{s}' for p in PROFILES for s in SCENARIOS]
TRBI_NAAQS_COLS = [f'TRBI_NAAQS_{p}_{s}' for p in PROFILES for s in SCENARIOS]

TIER1_TARGETS = [f'TRB_actual_{p}_mid' for p in PROFILES]
TIER2_TARGETS = [f'TRB_actual_{p}_low' for p in PROFILES]
TIER3_TARGETS = [f'TRB_actual_{p}_high' for p in PROFILES]
ALL_TARGETS = TRB_ACTUAL_COLS
REPR_TARGET = 'TRB_actual_m_s_mid'

# Physiological constants
VE_PROFILES = {'m_s': 0.0090, 'm_w': 0.0250, 'f_s': 0.0084, 'f_w': 0.0210}
F_TOTAL = {'pm2.5':0.995,'pm10':0.840,'so2':0.95,
           'no2':0.65,'o3':0.825,'co':0.875,'nh3':0.95}

# Reproducibility
FIG_DPI = 300
RANDOM_SEED = 42