# 🚀 Customer Churn Prediction Model

An industry-oriented Machine Learning project that predicts customer churn using behavioral, billing, and engagement data. This project demonstrates real-world Data Science, Business Analytics, and API deployment workflows used in telecom, SaaS, fintech, OTT, and subscription-based companies.

---

# 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services.

This project uses:

* Machine Learning
* XGBoost
* FastAPI
* Data Visualization
* Business Analytics

to build an end-to-end churn prediction system.

---

# 🎯 Objectives

* Predict customer churn probability
* Analyze customer behavior
* Improve retention strategies
* Build production-style ML workflow
* Deploy prediction API using FastAPI

---

# 🛠 Tech Stack

## Programming

* Python

## Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

## API

* FastAPI
* Uvicorn

## Tools

* Git
* GitHub
* VS Code

---

# 📂 Project Structure

```bash
Customer-Churn-Prediction/
│
├── data/
│   └── churn.csv
│
├── models/
│
├── outputs/
│
├── serving/
│   └── app.py
│
├── src/
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

Dataset Used:
IBM Telco Customer Churn Dataset

Features include:

* tenure
* monthly charges
* payment method
* internet service
* support usage
* customer demographics

Target:

* Churn (Yes/No)

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/bablu-c/customer-churn-prediction.git
```

## Navigate to Project

```bash
cd customer-churn-prediction
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

# ▶ Run Model Training

```bash
python src/train.py
```

---

# 🌐 Run FastAPI Server

```bash
uvicorn serving.app:app --reload
```

---

# 📌 Swagger API Docs

Open in browser:

```bash
http://127.0.0.1:8000/docs
```

---

# 📈 Model Performance

* Accuracy: ~77%
* ROC-AUC: ~0.81

Evaluation Metrics:

* Accuracy
* ROC-AUC
* Confusion Matrix
* Classification Report

---

# 📷 Outputs

The project generates:

* Confusion Matrix
* ROC Curve
* Feature Importance Chart
* API Prediction Results

---

# 🔥 Key Features

✅ Customer churn prediction
✅ XGBoost classifier
✅ Data preprocessing pipeline
✅ Visualization dashboard outputs
✅ FastAPI prediction API
✅ Business analytics workflow
✅ Production-style project structure

---

# 💼 Industry Relevance

This project simulates real-world churn analytics systems used in:

* Telecom
* SaaS
* OTT Platforms
* Fintech
* Subscription Businesses

---

# 🚀 Future Improvements

* SHAP Explainability
* Streamlit Dashboard
* Docker Deployment
* Cloud Deployment
* Automated Retraining
* Real-time Monitoring

---

# 👨‍💻 Author

Bablu kumar

---

# ⭐ If you found this project useful, consider giving it a star!
