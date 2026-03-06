"""Feature engineering for churn prediction."""
import pandas as pd
from src.utils.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-relevant derived features."""
    out = df.copy()
    if "Balance" in out.columns and "EstimatedSalary" in out.columns:
        out["BalanceToSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1)
    if "NumOfProducts" in out.columns:
        out["HasMultipleProducts"] = (out["NumOfProducts"] > 1).astype(int)
    if "Age" in out.columns and "Tenure" in out.columns:
        out["TenureToAgeRatio"] = out["Tenure"] / (out["Age"] + 1)
    return out


def get_feature_columns(
    df: pd.DataFrame,
    categorical: list = None,
    numeric: list = None,
    target: str = None,
) -> list:
    """Return list of feature column names (excluding target). Works before or after encoding."""
    target = target or TARGET_COLUMN
    if target not in df.columns:
        return list(df.columns)
    return [c for c in df.columns if c != target]
