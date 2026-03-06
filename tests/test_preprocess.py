"""Tests for data preprocessing."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate import validate_schema
from src.data.feature_engineering import add_derived_features, get_feature_columns
from src.data.preprocess import handle_missing, encode_categorical
from src.utils.config import TARGET_COLUMN, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _sample_df():
    return pd.DataFrame({
        "CreditScore": [650, 700],
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"],
        "Age": [40, 35],
        "Tenure": [5, 3],
        "Balance": [1000.0, 0.0],
        "NumOfProducts": [1, 2],
        "HasCrCard": [1, 0],
        "IsActiveMember": [1, 1],
        "EstimatedSalary": [50000.0, 60000.0],
        "Exited": [0, 1],
    })


def test_validate_schema():
    df = _sample_df()
    ok, err = validate_schema(df)
    assert ok is True
    assert err is None


def test_validate_schema_missing_col():
    df = _sample_df().drop(columns=["Exited"])
    ok, err = validate_schema(df)
    assert ok is False
    assert "Exited" in err or "Missing" in err


def test_add_derived_features():
    df = _sample_df()
    out = add_derived_features(df)
    assert "BalanceToSalaryRatio" in out.columns or "HasMultipleProducts" in out.columns


def test_handle_missing():
    df = _sample_df()
    out = handle_missing(df)
    assert out.isna().sum().sum() == 0


def test_encode_categorical():
    df = _sample_df()
    out, encoder = encode_categorical(df, fit=True)
    assert "Geography" not in out.columns or "Gender" not in out.columns
    assert len(out.columns) >= len(df.columns) - len(CATEGORICAL_COLUMNS)
