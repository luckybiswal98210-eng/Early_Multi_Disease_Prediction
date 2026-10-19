import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def train_diabetes_model():
    data_path = os.path.join("datasets", "diabetes.csv")
    df = pd.read_csv(data_path)
    
    print(f"Loaded Diabetes dataset with shape: {df.shape}")
    
    # Clinically, 0 values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are missing values
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_columns:
        median_val = df[df[col] > 0][col].median()
        df[col] = df[col].replace(0, median_val)
    
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Diabetes Model Accuracy: {acc * 100:.2f}% | ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "diabetes_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join("models", "diabetes_features.pkl"))
    print("Saved diabetes_model.pkl, diabetes_scaler.pkl, and diabetes_features.pkl")

if __name__ == "__main__":
    train_diabetes_model()
