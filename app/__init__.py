"""Flask application for Bank Churn Prediction."""
from flask import Flask
from pathlib import Path

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["ARTIFACTS_DIR"] = Path(__file__).resolve().parent.parent / "artifacts"
    app.config["SECRET_KEY"] = "churn-app-secret-change-in-production"
    from app.routes import register_routes
    register_routes(app)
    return app
