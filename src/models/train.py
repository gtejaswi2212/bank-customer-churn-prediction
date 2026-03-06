"""Model training pipeline: load data, preprocess, train multiple models, save best."""
import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils.config import (
    ARTIFACTS_DIR,
    ARTIFACTS_PLOTS,
    DEFAULT_DATA_PATH,
    METRICS_JSON,
    MODEL_FILENAME,
    SCALER_FILENAME,
    ENCODER_FILENAME,
    FEATURE_COLUMNS_JSON,
    RANDOM_STATE,
    USE_SMOTE,
    SMOTE_K_NEIGHBORS,
)
from src.utils.logger import get_logger
from src.data.preprocess import (
    load_and_preprocess,
    handle_missing,
    encode_categorical,
)
from src.data.feature_engineering import get_feature_columns
from src.models.evaluate import evaluate_model, METRIC_NAMES

logger = get_logger()

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def _prepare_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], StandardScaler, Any]:
    """Load, preprocess, encode, split, scale. Returns train/val/test and artifacts."""
    from src.utils.config import TARGET_COLUMN, TEST_SIZE, VAL_SIZE

    df = load_and_preprocess(data_path or DEFAULT_DATA_PATH)
    df = handle_missing(df)
    df, encoder = encode_categorical(df, fit=True)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=feature_cols,
        index=X_val.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    if USE_SMOTE and HAS_SMOTE:
        smote = SMOTE(k_neighbors=min(SMOTE_K_NEIGHBORS, sum(y_train == 1) - 1 or 1), random_state=RANDOM_STATE)
        try:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception as e:
            logger.warning("SMOTE failed: %s. Proceeding without oversampling.", e)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, encoder


def get_models() -> Dict[str, Any]:
    """Return dict of model name -> model instance."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    return models


def run_training(
    data_path: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    metric_for_best: str = "roc_auc",
) -> Dict[str, Any]:
    """Run full pipeline: prepare data, train all models, pick best, save artifacts."""
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_PLOTS.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, encoder = _prepare_data(data_path)

    models = get_models()
    results: List[Dict[str, Any]] = []
    best_name = None
    best_score = -np.inf
    best_model = None

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        metrics["model_name"] = name
        results.append(metrics)
        score = metrics.get(metric_for_best, metrics.get("roc_auc", 0))
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    # Final evaluation on test set
    test_metrics = evaluate_model(best_model, X_test, y_test)
    test_metrics["model_name"] = best_name

    # Save artifacts
    joblib.dump(best_model, artifacts_dir / MODEL_FILENAME)
    joblib.dump(scaler, artifacts_dir / SCALER_FILENAME)
    joblib.dump(encoder, artifacts_dir / ENCODER_FILENAME)
    with open(artifacts_dir / FEATURE_COLUMNS_JSON, "w") as f:
        json.dump(feature_cols, f, indent=2)

    all_metrics = {
        "best_model": best_name,
        "best_metric": metric_for_best,
        "best_score": best_score,
        "test_metrics": test_metrics,
        "model_comparison": results,
    }
    with open(artifacts_dir / METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Generate and save plots
    try:
        _save_plots(best_model, X_test, y_test, feature_cols, results, artifacts_dir)
    except Exception as e:
        logger.warning("Could not save plots: %s", e)

    logger.info("Best model: %s (val %s=%.4f)", best_name, metric_for_best, best_score)
    return all_metrics


def _save_plots(model, X_test, y_test, feature_cols, model_comparison, artifacts_dir: Path):
    """Save feature importance, confusion matrix, ROC curve, model comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Feature importance
    from src.models.explain import get_feature_importance
    imp = get_feature_importance(model, feature_cols)
    names = list(imp.keys())[:15]
    vals = [imp[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), vals, color="#2d5a87")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Churn Drivers)")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Churn", "Churn"])
    ax.set_yticklabels(["Not Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ROC curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#2d5a87", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Test Set)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "roc_curve.png", dpi=100, bbox_inches="tight")
        plt.close()

    # Model comparison (bar chart)
    names_comp = [r["model_name"] for r in model_comparison]
    roc_vals = [r.get("roc_auc", 0) for r in model_comparison]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names_comp, roc_vals, color=["#2d5a87", "#5a9bd4", "#87b8e0"])
    ax.set_ylabel("ROC-AUC (Validation)")
    ax.set_title("Model Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
