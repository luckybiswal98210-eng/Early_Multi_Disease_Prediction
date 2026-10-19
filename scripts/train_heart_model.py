import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def train_heart_model():
    data_path = os.path.join("datasets", "heart_disease.csv")
    df = pd.read_csv(data_path)
    
    print(f"Loaded Heart dataset with shape: {df.shape}")
    
    # Feature columns:
    # age, sex, cp (chest pain), trestbps (resting bp), chol, fbs (fasting blood sugar),
    # restecg, thalach (max HR), exang (exercise angina), oldpeak (ST depression),
    # slope, ca (major vessels), thal
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Heart Model Accuracy: {acc * 100:.2f}% | ROC-AUC: {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "heart_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "heart_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join("models", "heart_features.pkl"))
    print("Saved heart_model.pkl, heart_scaler.pkl, and heart_features.pkl")

if __name__ == "__main__":
    train_heart_model()
