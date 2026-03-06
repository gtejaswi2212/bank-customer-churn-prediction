"""Model explainability: feature importance and SHAP (optional)."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import ARTIFACTS_DIR, ARTIFACTS_PLOTS


def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from tree-based model or coefficients from linear model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
    else:
        return {name: 0.0 for name in feature_names}
    return dict(zip(feature_names, imp.tolist()))


def get_top_drivers(
    model: Any,
    feature_names: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-k churn drivers with names and importance."""
    imp = get_feature_importance(model, feature_names)
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:top_k]
    return [{"feature": name, "importance": round(float(val), 4)} for name, val in sorted_imp]


def get_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    max_display: int = 15,
) -> Optional[Dict[str, Any]]:
    """Compute SHAP values if shap is installed; otherwise return None."""
    try:
        import shap
    except ImportError:
        return None
    if hasattr(model, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(model, X) if hasattr(model, "feature_importances_") else shap.LinearExplainer(model, X)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            return {
                "shap_values": shap_vals.tolist() if hasattr(shap_vals, "tolist") else shap_vals,
                "feature_names": feature_names,
            }
        except Exception:
            return None
    return None
