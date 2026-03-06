"""Flask routes for churn prediction app."""
import json
from pathlib import Path

from flask import render_template, request, redirect, url_for, flash, send_from_directory

from src.models.predict import predict_churn


def register_routes(app):
    artifacts_dir = app.config.get("ARTIFACTS_DIR")
    metrics_path = (artifacts_dir / "metrics.json") if artifacts_dir else None
    plots_dir = (artifacts_dir / "plots") if artifacts_dir else None

    @app.route("/")
    def index():
        metrics = _load_metrics(metrics_path)
        return render_template("index.html", metrics=metrics)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            return render_template("predict.html", result=None)
        try:
            features = _form_to_features(request.form)
            result = predict_churn(features, artifacts_dir=artifacts_dir)
            return render_template("predict.html", result=result)
        except FileNotFoundError as e:
            flash("Model not found. Please run training first: python run_training.py", "warning")
            return render_template("predict.html", result=None)
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")
            return render_template("predict.html", result=None)

    @app.route("/insights")
    def insights():
        metrics = _load_metrics(metrics_path)
        has_plots = plots_dir and (plots_dir / "feature_importance.png").exists()
        return render_template(
            "insights.html",
            metrics=metrics,
            has_plots=has_plots,
            plots_dir=plots_dir,
        )

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/artifacts/plots/<path:filename>")
    def serve_plot(filename):
        if not plots_dir or not plots_dir.exists():
            return "Plots not found", 404
        return send_from_directory(plots_dir, filename)

    return app


def _load_metrics(metrics_path: Path):
    if not metrics_path or not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _form_to_features(form):
    """Map form fields to feature dict with correct types."""
    from src.utils.helpers import safe_float, safe_int
    return {
        "CreditScore": safe_int(form.get("CreditScore"), 650),
        "Geography": form.get("Geography", "France"),
        "Gender": form.get("Gender", "Female"),
        "Age": safe_int(form.get("Age"), 40),
        "Tenure": safe_int(form.get("Tenure"), 5),
        "Balance": safe_float(form.get("Balance"), 0.0),
        "NumOfProducts": safe_int(form.get("NumOfProducts"), 1),
        "HasCrCard": 1 if form.get("HasCrCard") in ("1", "on", "yes") else 0,
        "IsActiveMember": 1 if form.get("IsActiveMember") in ("1", "on", "yes") else 0,
        "EstimatedSalary": safe_float(form.get("EstimatedSalary"), 100000.0),
    }
