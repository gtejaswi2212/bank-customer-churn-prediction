"""Data loading and preprocessing pipeline."""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

from src.utils.config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_DATA_PATH,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data.validate import load_raw_data
from src.data.feature_engineering import add_derived_features, get_feature_columns


def load_and_preprocess(
    data_path: Optional[Path] = None,
    use_derived: bool = True,
) -> pd.DataFrame:
    """Load raw data, validate, optionally add derived features."""
    path = data_path or DEFAULT_DATA_PATH
    df = load_raw_data(path)
    if use_derived:
        df = add_derived_features(df)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    out = df.copy()
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in out.columns]
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())
    for c in CATEGORICAL_COLUMNS:
        if c in out.columns and out[c].isna().any():
            out[c] = out[c].fillna(out[c].mode().iloc[0] if len(out[c].mode()) else "Unknown")
    return out


def encode_categorical(
    df: pd.DataFrame,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode categorical columns."""
    from src.utils.config import CATEGORICAL_COLUMNS
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if not cat_cols:
        return df, encoder or OneHotEncoder(drop="first", sparse_output=False)

    X_cat = df[cat_cols].astype(str)
    if encoder is None:
        encoder = OneHotEncoder(drop="first", sparse_output=False)
    if fit:
        enc_arr = encoder.fit_transform(X_cat)
    else:
        enc_arr = encoder.transform(X_cat)

    enc_names = encoder.get_feature_names_out(cat_cols)
    enc_df = pd.DataFrame(enc_arr, columns=enc_names, index=df.index)
    out = df.drop(columns=cat_cols).join(enc_df)
    return out, encoder


def get_train_val_test_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split into train/val/test. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
