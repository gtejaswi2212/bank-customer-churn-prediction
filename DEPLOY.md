# Deploy Bank Churn Prediction

## Render (recommended)

### Option A: One-click with Blueprint

1. Go to [dashboard.render.com](https://dashboard.render.com).
2. Click **New** → **Blueprint**.
3. Connect your GitHub account and select the repo `bank-customer-churn-prediction`.
4. Render will read `render.yaml` and create a Web Service with:
   - **Build:** `pip install -r requirements.txt && python run_training.py`
   - **Start:** `gunicorn --bind 0.0.0.0:$PORT run:app`
5. Deploy. Your app will be at `https://<service-name>.onrender.com`.

### Option B: Manual Web Service

1. **New** → **Web Service**.
2. Connect the GitHub repo.
3. Set:
   - **Build Command:** `pip install -r requirements.txt && python run_training.py`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT run:app`
4. **Create Web Service**.

### If build fails (e.g. out of memory)

Free tier has limited RAM. Train locally and commit artifacts:

```bash
python run_training.py
git add artifacts/
git commit -m "Add trained artifacts for deploy"
git push
```

Then in Render, set **Build Command** to only:

```bash
pip install -r requirements.txt
```

---

## Railway

1. Go to [railway.app](https://railway.app) and **New Project** → **Deploy from GitHub**.
2. Select `bank-customer-churn-prediction`.
3. Railway uses the **Procfile** by default: `web: gunicorn --bind 0.0.0.0:$PORT run:app`.
4. Add a **Build** step: in Settings, set build command to `pip install -r requirements.txt && python run_training.py`, or commit `artifacts/` and use only `pip install -r requirements.txt`.

---

## Environment

- `PORT` is set automatically by Render/Railway; no need to configure.
- Optional: set `FLASK_ENV=production` if you add env-based config later.
