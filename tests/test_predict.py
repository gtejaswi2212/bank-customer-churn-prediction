"""Tests for prediction helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import get_risk_category, get_retention_action


def test_risk_category_low():
    assert get_risk_category(0.2) == "Low"
    assert get_risk_category(0.34) == "Low"


def test_risk_category_medium():
    assert get_risk_category(0.5) == "Medium"
    assert get_risk_category(0.35) == "Medium"
    assert get_risk_category(0.65) == "Medium"


def test_risk_category_high():
    assert get_risk_category(0.7) == "High"
    assert get_risk_category(0.66) == "High"


def test_retention_action():
    assert "retention" in get_retention_action(0.8, "High").lower() or "outreach" in get_retention_action(0.8, "High").lower()
    assert get_retention_action(0.2, "Low")
