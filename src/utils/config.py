"""Central configuration for the churn prediction pipeline."""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_PLOTS = ARTIFACTS_DIR / "plots"

# Ensure directories exist
for d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS_DIR, ARTIFACTS_PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# Dataset
DEFAULT_DATA_PATH = PROJECT_ROOT / "analytical_base_table.csv"
TARGET_COLUMN = "Exited"

# Categorical and numeric columns (must match dataset schema)
CATEGORICAL_COLUMNS = ["Geography", "Gender"]
NUMERIC_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Train/val/test split
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # of remaining after test
RANDOM_STATE = 42

# Class imbalance
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 5

# Model artifacts
MODEL_FILENAME = "model.pkl"
SCALER_FILENAME = "scaler.pkl"
ENCODER_FILENAME = "encoder.pkl"
FEATURE_COLUMNS_JSON = "feature_columns.json"
METRICS_JSON = "metrics.json"

# Churn risk thresholds (probability)
RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.65
