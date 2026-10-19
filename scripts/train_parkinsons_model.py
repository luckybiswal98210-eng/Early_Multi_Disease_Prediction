import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- PATHS ----------
BASE_AUDIO_DIR = os.path.join("data", "audio")   # data/audio
LABELS_CSV = os.path.join("data", "labels.csv")  # filename,label

# ---------- LOAD LABELS ----------
labels_df = pd.read_csv(LABELS_CSV)

# filename column must contain relative paths like:
#   HC_AH/AH_....wav
#   PD_AH/AH_....wav
print("Loaded labels:", labels_df.shape)
print(labels_df.head())

def extract_features(path, sr_target=22050, n_mfcc=20):
    """Extract simple features from one wav file."""
    y, sr = librosa.load(path, sr=sr_target)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()

    # Root-mean-square energy
    rms = librosa.feature.rms(y=y)[0].mean()

    # Spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()

    # Combine all features
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])

X_list, y_list = [], []

for idx, row in labels_df.iterrows():
    rel_path = row["filename"]   # e.g. "HC_AH/AH_....wav"
    label = row["label"]         # 0 or 1

    full_path = os.path.join(BASE_AUDIO_DIR, rel_path)

    if not os.path.exists(full_path):
        print("Missing file, skipping:", full_path)
        continue

    try:
        feats = extract_features(full_path)
        X_list.append(feats)
        y_list.append(label)
    except Exception as e:
        print("Error processing", full_path, ":", e)

X = np.vstack(X_list)
y = np.array(y_list)

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- SCALER + MODEL ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy on held-out test set: {:.2f}%".format(acc * 100))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

# ---------- SAVE MODEL ----------
joblib.dump(model, "audio_parkinsons_model.pkl")
joblib.dump(scaler, "audio_scaler.pkl")
print("\nSaved audio_parkinsons_model.pkl and audio_scaler.pkl")
