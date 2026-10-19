import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def train_kidney_model():
    data_path = os.path.join("datasets", "kidney_disease.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded Chronic Kidney Disease dataset with shape: {df.shape}")
    
    # Target normalization
    df["classification"] = df["classification"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})
    df = df.dropna(subset=["classification"])
    
    # Select high-impact clinical features
    # age, bp, sg (specific gravity), al (albumin), su (sugar), bgr (blood glucose random),
    # bu (blood urea), sc (serum creatinine), hemo (hemoglobin), htn (hypertension), dm (diabetes)
    cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo", "htn", "dm"]
    
    df["htn"] = df["htn"].astype(str).str.strip().map({"yes": 1, "no": 0}).fillna(0)
    df["dm"] = df["dm"].astype(str).str.strip().map({"yes": 1, "no": 0, "\tyes": 1, " yes": 1}).fillna(0)
    
    for c in ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
        
    X = df[cols]
    y = df["classification"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
    
    print(f"Kidney Model Test Accuracy: {acc * 100:.2f}% | ROC-AUC: {auc:.4f}")
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "kidney_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "kidney_scaler.pkl"))
    joblib.dump(cols, os.path.join("models", "kidney_features.pkl"))
    print("Saved kidney_model.pkl, kidney_scaler.pkl, and kidney_features.pkl")

if __name__ == "__main__":
    train_kidney_model()
