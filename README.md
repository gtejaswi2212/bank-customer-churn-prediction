# Bank Customer Churn Prediction

A production-style, end-to-end **customer churn prediction** project: from raw data to a deployable web app with model insights and business-oriented outputs. Built to showcase ML fundamentals, software engineering practices, and clear business impact.

---

## Overview

**Business problem:** Banks need to identify customers at risk of churning so they can take proactive retention actions. This project delivers a trained classifier, explainability (feature importance, risk categories), and a web interface for predictions and insights.

**What’s included:**

- **Modular ML pipeline:** load → validate → preprocess → encode → scale → train (Logistic Regression, Random Forest, XGBoost) → evaluate → save artifacts
- **Reproducible training** with train/validation/test splits, optional SMOTE for class imbalance, and saved encoders/scalers
- **Model explainability:** feature importance plots, confusion matrix, ROC curve, model comparison
- **Web app:** landing page, prediction form (churn probability + risk category + retention action), model insights, about page
- **Tests** for preprocessing and prediction helpers
- **Deployment-ready:** Gunicorn, Procfile, environment-friendly config

---

## Dataset

- **File:** `analytical_base_table.csv` (in repo root; can also be placed under `data/raw/`)
- **Rows:** ~10,000 customers
- **Columns:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, **Exited** (churn target)

---

## Project Structure

```
bank-customer-churn-prediction/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # Routes and form handling
│   ├── templates/           # Jinja2 HTML
│   └── static/css/          # Styles
├── src/
│   ├── data/
│   │   ├── preprocess.py    # Load, encode, split
│   │   ├── validate.py      # Schema validation
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py         # Full training pipeline
│   │   ├── predict.py       # Inference + risk + actions
│   │   ├── evaluate.py      # Metrics
│   │   └── explain.py       # Feature importance / SHAP
│   └── utils/
│       ├── config.py        # Paths, constants
│       ├── logger.py
│       └── helpers.py
├── notebooks/
│   └── churn_eda.ipynb      # EDA
├── artifacts/               # After training
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── feature_columns.json
│   ├── metrics.json
│   └── plots/               # feature_importance, confusion_matrix, roc_curve, model_comparison
├── tests/
│   ├── test_preprocess.py
│   └── test_predict.py
├── data/
│   ├── raw/
│   └── processed/
├── run.py                   # Flask entry
├── run_training.py          # Train and save artifacts
├── requirements.txt
├── Procfile                 # e.g. Render / Railway
└── README.md
```

---

## Setup

1. **Clone and install**

   ```bash
   git clone https://github.com/gtejaswi2212/Bank-Churn-Prediction.git
   cd Bank-Churn-Prediction
   pip install -r requirements.txt
   ```

2. **Train the model** (creates `artifacts/` and plots)

   ```bash
   python run_training.py
   ```

3. **Run the web app**

   ```bash
   python run.py
   ```

   Open **http://127.0.0.1:5000**. You’ll see the landing page, **Try Prediction**, **Model Insights**, and **About**.

---

## How to Run Locally

| Step | Command |
|------|--------|
| Install deps | `pip install -r requirements.txt` |
| Train model | `python run_training.py` |
| Start app | `python run.py` |
| Run tests | `pytest tests/` |

The app expects `artifacts/model.pkl`, `scaler.pkl`, `encoder.pkl`, and `feature_columns.json`. If they’re missing, run `python run_training.py` first; the prediction page will show a short message if the model isn’t found.

---

## ML Workflow

1. **Data:** Load `analytical_base_table.csv`, validate schema.
2. **Preprocessing:** Handle missing values, one-hot encode Geography/Gender, optional derived features, StandardScaler (fit on train).
3. **Split:** Train / validation / test (stratified).
4. **Imbalance:** SMOTE on training set (configurable).
5. **Models:** Logistic Regression, Random Forest, XGBoost; best by validation ROC-AUC.
6. **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix on test set.
7. **Artifacts:** Best model, scaler, encoder, feature list, metrics JSON, and plots under `artifacts/plots/`.

---

## Web App Features

- **Landing:** Project intro, key metrics (if trained), workflow, tech stack, challenges & learnings, CTAs to Try Prediction, Model Insights, GitHub.
- **Prediction:** Form with all customer fields → churn prediction, probability, **risk category** (Low / Medium / High), short explanation, **recommended retention action**.
- **Model Insights:** Test metrics, feature importance, confusion matrix, ROC curve, model comparison, and short business insights.
- **About:** Dataset, approach, preprocessing, modeling, evaluation, future improvements.

---

## Deploy to Render (free tier)

1. **Push this repo to GitHub** (you already have it at `github.com/gtejaswi2212/bank-customer-churn-prediction`).

2. **Go to [Render Dashboard](https://dashboard.render.com/)** → **New** → **Web Service**.

3. **Connect the repo** and select `bank-customer-churn-prediction`.

4. **Settings:**
   - **Build Command:** `pip install -r requirements.txt && python run_training.py`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT run:app`
   - **Plan:** Free

5. Click **Create Web Service**. The first deploy will install deps, train the model, and start the app. Your app will be live at `https://<your-service>.onrender.com`.

**Alternative (Blueprint):** If your repo has `render.yaml`, use **New** → **Blueprint** and connect the same repo; Render will read build/start commands from the file.

**If the build runs out of memory** (training on free tier can be tight): run `python run_training.py` locally, commit the `artifacts/` folder, then set **Build Command** to just `pip install -r requirements.txt`.

---

## Future Improvements

- Hyperparameter tuning (e.g. GridSearchCV / Optuna)
- SHAP in the app for per-prediction explanations
- Scheduled retraining or pipeline on new data
- Docker image and one-click deploy

---

## Author

**Tejaswi Ganji**  
📧 [Email](mailto:tejaswi.ganji2000@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/gtejaswi2212) | 💻 [GitHub](https://github.com/gtejaswi2212)

---

If you found this useful, consider giving it a ⭐ on GitHub.
