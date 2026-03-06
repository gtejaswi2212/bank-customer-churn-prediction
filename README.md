# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)
![Deployment](https://img.shields.io/badge/Deployment-Render-green)

🔗 **Live Demo:**
[https://bank-customer-churn-prediction-9k9h.onrender.com/](https://bank-customer-churn-prediction-9k9h.onrender.com/)

An **end-to-end machine learning system for predicting customer churn** with a deployable web application.
This project demonstrates how predictive models can be integrated into production-style systems to help businesses identify customers likely to leave and take proactive retention actions.

The system includes a **complete ML pipeline, explainability, a prediction interface, and cloud deployment**.

---

# 🚀 Project Impact

This project demonstrates how machine learning can support **customer retention strategies** in financial services.

Key outcomes:

• Built a **modular ML pipeline** from raw data to deployment
• Compared **multiple machine learning models**
• Implemented **model explainability using feature importance analysis**
• Developed a **web application for real-time churn predictions**
• Added **business-oriented churn risk categories and retention suggestions**
• Deployed the application to the cloud for public access

---

# 🌐 Live Application

Try the deployed application:

👉 [https://bank-customer-churn-prediction-9k9h.onrender.com/](https://bank-customer-churn-prediction-9k9h.onrender.com/)

Features available in the live demo:

• Real-time churn prediction
• Probability score for churn risk
• Risk classification (Low / Medium / High)
• Recommended retention actions
• Model performance insights

---

# 🧠 Problem Statement

Customer churn significantly impacts profitability for subscription-based and financial services businesses.

Research shows acquiring a new customer can cost **5–7× more than retaining an existing one**.

Predictive models help businesses identify **high-risk customers early**, enabling targeted retention strategies.

This project builds a machine learning system that predicts whether a customer will churn based on demographic and financial attributes.

---

# 📊 Dataset

The dataset contains approximately **10,000 bank customers** with demographic and financial attributes.

| Feature         | Description                  |
| --------------- | ---------------------------- |
| CreditScore     | Customer credit score        |
| Geography       | Country of residence         |
| Gender          | Customer gender              |
| Age             | Age of customer              |
| Tenure          | Years with the bank          |
| Balance         | Account balance              |
| NumOfProducts   | Number of bank products used |
| HasCrCard       | Credit card ownership        |
| IsActiveMember  | Activity status              |
| EstimatedSalary | Estimated income             |
| Exited          | Target variable (1 = churn)  |

---

# ⚙️ Machine Learning Pipeline

The model training workflow includes:

### 1️⃣ Data Validation

Schema validation and dataset verification.

### 2️⃣ Data Preprocessing

• Missing value handling
• One-hot encoding for categorical variables
• Feature scaling with StandardScaler

### 3️⃣ Train/Test Split

Stratified splitting to maintain class balance.

### 4️⃣ Handling Class Imbalance

SMOTE applied to improve minority class prediction.

### 5️⃣ Model Training

Algorithms evaluated:

• Logistic Regression
• Random Forest
• XGBoost

### 6️⃣ Model Evaluation

Metrics used:

• Accuracy
• Precision
• Recall
• F1 Score
• ROC-AUC

Generated visualizations:

• Feature importance
• Confusion matrix
• ROC curve
• Model comparison

### 7️⃣ Deployment

The best performing model is serialized and integrated into a Flask web application for real-time predictions.

---

# 🏗 System Architecture

```
User Input (Web Form)
        │
        ▼
Flask Web Application
        │
        ▼
Prediction Pipeline
        │
        ├── Data Preprocessing
        ├── Feature Encoding
        ├── Feature Scaling
        │
        ▼
Trained ML Model
        │
        ▼
Prediction Output
        │
        ├── Churn Probability
        ├── Risk Category
        └── Retention Recommendation
```

---

# 💻 Web Application

The project includes a **fully functional web interface**.

### Landing Page

Overview of the project and navigation.

### Prediction Page

Users can input customer information to receive:

• churn prediction
• churn probability score
• risk category
• retention recommendations

### Model Insights Page

Displays:

• feature importance visualization
• confusion matrix
• ROC curve
• model comparison results

### About Page

Explains the dataset, methodology, and system architecture.

---

# 📸 Screenshots

Add screenshots of your deployed application.

### Landing Page

![Landing Page](images/landing_page.png)

### Prediction Page

![Prediction](images/prediction_page.png)

### Model Insights

![Insights](images/model_insights.png)

Create a folder:

```
images/
landing_page.png
prediction_page.png
model_insights.png
```

---

# 📁 Project Structure

```
bank-customer-churn-prediction
│
├── app
│   ├── routes.py
│   ├── templates
│   └── static
│
├── src
│   ├── data
│   ├── models
│   └── utils
│
├── notebooks
│   └── churn_eda.ipynb
│
├── artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── plots
│
├── tests
│
├── data
│   ├── raw
│   └── processed
│
├── run.py
├── run_training.py
├── requirements.txt
└── README.md
```

---

# 🛠 Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/gtejaswi2212/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python run_training.py
```

### Run the web application

```bash
python run.py
```

Open the app:

```
http://127.0.0.1:5000
```

---

# ☁ Deployment

The application is deployed on **Render**.

Build Command

```
pip install -r requirements.txt && python run_training.py
```

Start Command

```
gunicorn --bind 0.0.0.0:$PORT run:app
```

---

# 📈 Business Use Case

Customer success teams can use this system to identify customers with high churn probability and take proactive retention actions.

Possible strategies:

• loyalty incentives
• personalized engagement campaigns
• proactive support outreach

Early churn detection can help businesses improve **customer lifetime value and retention rates**.

---

# 🔮 Future Improvements

• SHAP-based explainability
• automated model retraining
• API endpoints for batch predictions
• Docker containerization
• interactive dashboards

---

# 👩‍💻 Author

**Tejaswi Ganji**

MS Data Science — Stony Brook University

📧 [tejaswi.ganji2000@gmail.com](mailto:tejaswi.ganji2000@gmail.com)
💼 [https://linkedin.com/in/gtejaswi2212](https://linkedin.com/in/gtejaswi2212)
💻 [https://github.com/gtejaswi2212](https://github.com/gtejaswi2212)

---

⭐ If you found this project useful, consider giving it a **star** on GitHub.
