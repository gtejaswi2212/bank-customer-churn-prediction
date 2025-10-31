# 🏦 Bank Churn Prediction  

**Tools:** Python, Pandas, Scikit-learn, XGBoost, Flask  
**Dataset:** `analytical_base_table.csv`  

---

## 📊 Overview  
This project aims to predict **customer churn** for a bank using machine learning models.  
By analyzing customer demographics and transaction history, the model identifies customers who are most likely to leave, helping the bank take proactive retention actions.  

It includes:
- **Exploratory Data Analysis (EDA)** to uncover behavioral patterns  
- **Model training and optimization** using Scikit-learn and XGBoost  
- **Flask web app deployment** for real-time prediction  

---

## 🧠 Objectives  
- Build a predictive model that classifies customers as *churn* or *non-churn*.  
- Identify the main drivers behind customer churn.  
- Deploy the model as a simple web application for real-time inference.  

---

## 📈 Project Workflow  

### 1️⃣ Data Preparation  
- Imported and cleaned `analytical_base_table.csv`  
- Handled missing values and outliers  
- Encoded categorical variables and scaled numeric features  
- Balanced the dataset using oversampling  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Examined churn distribution and customer demographics  
- Visualized key features such as tenure, balance, and credit score  
- Identified correlations between churn and customer attributes  

### 3️⃣ Model Building  
- Built and compared multiple models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost (final model)  
- Optimized hyperparameters using **GridSearchCV**  
- Evaluated models based on:
  - Accuracy  
  - Precision, Recall, F1-Score  
  - ROC-AUC  

### 4️⃣ Model Deployment  
- Built a **Flask web app** (`bank_churn_prediction.py`) to serve model predictions.  
- Integrated trained XGBoost model for inference.  
- Configured **Gunicorn** for WSGI deployment and scalability.  


---

## ⚙️ Installation & Setup  

1️⃣ Clone the Repository
```bash
git clone https://github.com/gtejaswi2212/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the Flask Application
```bash
python bank_churn_prediction.py
```
---

🧰 Key Libraries

- pandas, numpy → Data processing and analysis
- scikit-learn → ML model building and evaluation
- xgboost → Boosted tree model for superior accuracy
- imbalanced-learn → Oversampling (SMOTE) for class balance
- Flask, Gunicorn → Backend web framework and deployment

---

💡 Insights & Takeaways

- Customers with low tenure, low balance, and high credit card usage show higher churn rates.
- Retention strategies should focus on offering loyalty benefits to these customers.
- The XGBoost model achieved the highest ROC-AUC score and balanced precision-recall.

---

🧾 Requirements

All dependencies are listed in requirements.txt.
Install them using:
```bash
pip install -r requirements.txt
```
---

🌐 Future Enhancements

- Deploy the model using Docker or AWS Lambda for scalability.
- Integrate Streamlit dashboard for visual interaction and prediction testing.
- Automate data refresh and retraining for real-time churn tracking.

---


## 👤 Author  
**Tejaswi Ganji**  
📧 [Email](mailto:tejaswi.ganji2000@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/gtejaswi2212) | 💻 [GitHub](https://github.com/gtejaswi2212)  

---

⭐ **If you found this project insightful, give it a star on GitHub!**
