#!/usr/bin/env python3
"""Run the full ML pipeline: preprocess, train, evaluate, save artifacts and plots."""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import run_training

if __name__ == "__main__":
    metrics = run_training()
    print("Training complete. Best model:", metrics["best_model"])
    print("Test metrics:", metrics["test_metrics"])
