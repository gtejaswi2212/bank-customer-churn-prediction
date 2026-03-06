"""Data validation and schema checks."""
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.utils.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Check that dataframe has required columns and types."""
    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' should be numeric"
    return True, None


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load CSV and validate schema."""
    df = pd.read_csv(path)
    ok, err = validate_schema(df)
    if not ok:
        raise ValueError(f"Invalid data: {err}")
    return df
