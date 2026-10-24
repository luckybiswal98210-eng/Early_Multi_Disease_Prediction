import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# Target artifact directory
ARTIFACT_DIR = "/Users/luckybiswal/.gemini/antigravity-ide/brain/3f5cfbf2-ef87-4c62-870d-26485733a265"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------------------------------------
# PLOT 1: MODEL PERFORMANCE COMPARISON CHART
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
modules = ["Parkinson's (Voice)", "Heart Disease (CAD)", "Diabetes Mellitus", "Chronic Kidney (CKD)"]
accs = [85.2, 83.61, 73.38, 98.75]
aucs = [91.2, 91.56, 80.81, 99.80]
recalls = [84.0, 85.0, 67.0, 98.0]
f1s = [85.0, 84.0, 69.0, 99.0]

x = np.arange(len(modules))
width = 0.2

rects1 = ax.bar(x - 1.5*width, accs, width, label='Test Accuracy (%)', color='#0284c7', edgecolor='#0369a1')
rects2 = ax.bar(x - 0.5*width, aucs, width, label='ROC-AUC Score (x100)', color='#10b981', edgecolor='#047857')
rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall / Sensitivity (%)', color='#f59e0b', edgecolor='#d97706')
rects4 = ax.bar(x + 1.5*width, f1s, width, label='F1-Score (%)', color='#8b5cf6', edgecolor='#6d28d9')

ax.set_ylabel('Percentage / Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Clinical Disease Prediction System — Machine Learning Evaluation Benchmarks', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(modules, fontsize=11, fontweight='bold')
ax.set_ylim(50, 105)
ax.axhline(80, color='#94a3b8', linestyle='--', linewidth=0.8, alpha=0.7)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(axis='y', linestyle=':', alpha=0.6)

for rect in rects1 + rects2 + rects3 + rects4:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=45)

plt.tight_layout()
p1_path = os.path.join(ARTIFACT_DIR, "model_training_comparison.png")
plt.savefig(p1_path)
plt.close()
print("Saved:", p1_path)

# -------------------------------------------------------------
# PLOT 2: CONFUSION MATRICES & FEATURE IMPORTANCES
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)

# Load test models and datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Heart
df_h = pd.read_csv("datasets/heart_disease.csv")
X_h = df_h.drop(columns=["target"])
y_h = df_h["target"]
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
scaler_h = joblib.load("models/heart_scaler.pkl")
model_h = joblib.load("models/heart_model.pkl")
y_pred_h = model_h.predict(scaler_h.transform(X_te_h))
cm_h = confusion_matrix(y_te_h, y_pred_h)

sns.heatmap(cm_h, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
            xticklabels=['No Disease', 'Heart Disease'], yticklabels=['No Disease', 'Heart Disease'])
axes[0, 0].set_title("Heart Disease Confusion Matrix\n(Accuracy: 83.61%)", fontweight='bold')
axes[0, 0].set_ylabel('True Clinical Diagnosis', fontweight='bold')
axes[0, 0].set_xlabel('Predicted Diagnosis', fontweight='bold')

# Diabetes
df_d = pd.read_csv("datasets/diabetes.csv")
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    med = df_d[df_d[col] > 0][col].median()
    df_d[col] = df_d[col].replace(0, med)
X_d = df_d.drop(columns=["Outcome"])
y_d = df_d["Outcome"]
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42, stratify=y_d)
scaler_d = joblib.load("models/diabetes_scaler.pkl")
model_d = joblib.load("models/diabetes_model.pkl")
y_pred_d = model_d.predict(scaler_d.transform(X_te_d))
cm_d = confusion_matrix(y_te_d, y_pred_d)

sns.heatmap(cm_d, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1], cbar=False,
            xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
axes[0, 1].set_title("Diabetes Confusion Matrix\n(Accuracy: 73.38%)", fontweight='bold')
axes[0, 1].set_ylabel('True Clinical Diagnosis', fontweight='bold')
axes[0, 1].set_xlabel('Predicted Diagnosis', fontweight='bold')

# Kidney
df_k = pd.read_csv("datasets/kidney_disease.csv")
df_k["classification"] = df_k["classification"].astype(str).str.strip().map({"ckd": 1, "notckd": 0})
df_k = df_k.dropna(subset=["classification"])
cols_k = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo", "htn", "dm"]
df_k["htn"] = df_k["htn"].astype(str).str.strip().map({"yes": 1, "no": 0}).fillna(0)
df_k["dm"] = df_k["dm"].astype(str).str.strip().map({"yes": 1, "no": 0, "\tyes": 1, " yes": 1}).fillna(0)
for c in ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "hemo"]:
    df_k[c] = pd.to_numeric(df_k[c], errors="coerce").fillna(df_k[c].median())
X_k = df_k[cols_k]
y_k = df_k["classification"].astype(int)
X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(X_k, y_k, test_size=0.2, random_state=42, stratify=y_k)
scaler_k = joblib.load("models/kidney_scaler.pkl")
model_k = joblib.load("models/kidney_model.pkl")
y_pred_k = model_k.predict(scaler_k.transform(X_te_k))
cm_k = confusion_matrix(y_te_k, y_pred_k)

sns.heatmap(cm_k, annot=True, fmt='d', cmap='Greens', ax=axes[0, 2], cbar=False,
            xticklabels=['No CKD', 'CKD Stage 1-5'], yticklabels=['No CKD', 'CKD Stage 1-5'])
axes[0, 2].set_title("Chronic Kidney Confusion Matrix\n(Accuracy: 100.0% / CV: 98.75%)", fontweight='bold')
axes[0, 2].set_ylabel('True Clinical Diagnosis', fontweight='bold')
axes[0, 2].set_xlabel('Predicted Diagnosis', fontweight='bold')

# Row 2: Top Feature Importances for each
# Heart top features
fi_h = pd.Series(model_h.feature_importances_, index=X_h.columns).sort_values(ascending=True).tail(6)
fi_h.plot(kind='barh', ax=axes[1, 0], color='#0284c7', edgecolor='#0f172a')
axes[1, 0].set_title("Top 6 Biomarkers: Heart Disease", fontweight='bold')
axes[1, 0].set_xlabel("Gini Feature Importance")

# Diabetes top features
fi_d = pd.Series(model_d.feature_importances_, index=X_d.columns).sort_values(ascending=True).tail(6)
fi_d.plot(kind='barh', ax=axes[1, 1], color='#f59e0b', edgecolor='#0f172a')
axes[1, 1].set_title("Top 6 Biomarkers: Diabetes", fontweight='bold')
axes[1, 1].set_xlabel("Gini Feature Importance")

# Kidney top features
fi_k = pd.Series(model_k.feature_importances_, index=X_k.columns).sort_values(ascending=True).tail(6)
fi_k.plot(kind='barh', ax=axes[1, 2], color='#10b981', edgecolor='#0f172a')
axes[1, 2].set_title("Top 6 Biomarkers: Kidney Disease", fontweight='bold')
axes[1, 2].set_xlabel("Gini Feature Importance")

plt.tight_layout()
p2_path = os.path.join(ARTIFACT_DIR, "confusion_matrices_and_features.png")
plt.savefig(p2_path)
plt.close()
print("Saved:", p2_path)
