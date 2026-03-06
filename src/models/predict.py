"""Inference: load artifacts and predict churn with risk category and business message."""
import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    ARTIFACTS_DIR,
    ENCODER_FILENAME,
    FEATURE_COLUMNS_JSON,
    MODEL_FILENAME,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    SCALER_FILENAME,
)
from src.data.feature_engineering import add_derived_features
from src.data.preprocess import handle_missing, encode_categorical


def load_artifacts(artifacts_dir: Optional[Path] = None) -> Tuple[Any, Any, Any, List[str]]:
    """Load model, scaler, encoder, and feature columns."""
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    model = joblib.load(artifacts_dir / MODEL_FILENAME)
    scaler = joblib.load(artifacts_dir / SCALER_FILENAME)
    encoder = joblib.load(artifacts_dir / ENCODER_FILENAME)
    with open(artifacts_dir / FEATURE_COLUMNS_JSON, "r") as f:
        feature_columns = json.load(f)
    return model, scaler, encoder, feature_columns


def _dataframe_from_input(features: Dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame from form/API input."""
    # Map form keys to schema; ensure types
    column_defaults = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100000.0,
    }
    row = {k: features.get(k, v) for k, v in column_defaults.items()}
    return pd.DataFrame([row])


def _preprocess_single(
    df: pd.DataFrame,
    encoder: Any,
    scaler: Any,
    feature_columns: List[str],
) -> np.ndarray:
    """Add derived, handle missing, encode, scale; return array with feature_columns order."""
    df = add_derived_features(df)
    df = handle_missing(df)
    df, _ = encode_categorical(df, encoder=encoder, fit=False)
    # Align columns (missing columns filled with 0 for one-hot unseen categories)
    X = pd.DataFrame(0.0, index=df.index, columns=feature_columns)
    for c in feature_columns:
        if c in df.columns:
            X[c] = df[c].values
    X = scaler.transform(X)
    return X


def get_risk_category(probability: float) -> str:
    if probability < RISK_LOW_THRESHOLD:
        return "Low"
    if probability > RISK_HIGH_THRESHOLD:
        return "High"
    return "Medium"


def get_retention_action(probability: float, risk_category: str) -> str:
    if risk_category == "Low":
        return "No immediate action required. Continue standard engagement."
    if risk_category == "Medium":
        return "Consider proactive outreach: loyalty offer or satisfaction check."
    return "Elevated churn risk; retention outreach recommended (e.g., personalized offer or retention call)."


def predict_churn(
    features: Dict[str, Any],
    artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Predict churn and return probability, risk category, and business message."""
    model, scaler, encoder, feature_columns = load_artifacts(artifacts_dir)
    df = _dataframe_from_input(features)
    X = _preprocess_single(df, encoder, scaler, feature_columns)

    proba = model.predict_proba(X)[0, 1]
    predicted = 1 if proba >= 0.5 else 0
    risk = get_risk_category(proba)
    action = get_retention_action(proba, risk)

    return {
        "churn_prediction": int(predicted),
        "churn_probability": round(float(proba), 4),
        "risk_category": risk,
        "retention_action": action,
        "explanation": _short_explanation(proba, risk),
    }


def _short_explanation(probability: float, risk_category: str) -> str:
    if risk_category == "Low":
        return "Customer shows low likelihood to churn based on current profile."
    if risk_category == "High":
        return "This customer has elevated churn risk; retention outreach recommended."
    return "Moderate churn risk; consider monitoring and light-touch engagement."
