import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from xgboost import XGBClassifier


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("data/churn.csv")

print("\nDataset Preview:\n")
print(df.head())


# ==============================
# REMOVE UNNECESSARY COLUMNS
# ==============================

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)


# ==============================
# HANDLE MISSING VALUES
# ==============================

df.replace(" ", np.nan, inplace=True)

df.dropna(inplace=True)


# ==============================
# ENCODE CATEGORICAL COLUMNS
# ==============================

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])


# ==============================
# FEATURES & TARGET
# ==============================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# MODEL TRAINING
# ==============================

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ==============================
# PREDICTIONS
# ==============================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_test, predictions)

roc_auc = roc_auc_score(y_test, probabilities)

print("\nAccuracy Score:")
print(accuracy)

print("\nROC-AUC Score:")
print(roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.show()


# ==============================
# ROC CURVE
# ==============================

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.savefig("outputs/roc_curve.png")

plt.show()


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(10)
)

plt.title("Top 10 Important Features")

plt.savefig("outputs/feature_importance.png")

plt.show()


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "models/churn_model.pkl")

print("\nModel Saved Successfully!")


# ==============================
# SAMPLE PREDICTIONS
# ==============================

sample_results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": predictions[:10],
    "Probability": probabilities[:10]
})

print("\nSample Predictions:\n")

print(sample_results)