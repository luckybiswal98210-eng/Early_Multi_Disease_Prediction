# 🩺 Intelligent Healthcare Monitoring & Early Disease Prediction System
**Semester 5 Engineering Design Project**

An end-to-end, multi-disease Clinical Decision Support System (CDSS) built using Machine Learning and Audio Signal Processing. The system provides early risk screening across **four critical physiological systems** via a single, unified web application — requiring **zero external hardware**.

---

## 🌟 Key Highlights

- **4-in-1 Disease Coverage:**
  1. 🎙️ **Parkinson's Disease:** Acoustic phonation analysis (MFCCs, ZCR, RMS, Spectral Centroid) + 5-point motor symptom checklist + 13-language spoken diagnosis.
  2. ❤️ **Cardiovascular (Heart) Disease:** Coronary artery disease risk estimation using hemodynamic vitals and stress test biomarkers.
  3. 🩸 **Diabetes Mellitus:** Metabolic dysfunction and insulin resistance risk assessment based on glycemic and anthropometric metrics.
  4. 🫘 **Chronic Kidney Disease (CKD):** Early asymptomatic renal failure detection using routine blood/urine biochemical filtration markers.
- **🤖 AI Medical Chatbot & Visual Image Triaging:**
  - **Conversational Symptom Checker:** Interactive triage asking targeted follow-up questions for acute symptoms (e.g., fever duration/temperature, cough, acidity, headaches).
  - **Evidence-Based Home Remedies:** Instant actionable herbal and supportive home remedies with clinical red flag warnings.
  - **Medical Image Disease Identifier:** Uploads medical photos or scans (edema photos, ECG strips, glucometer screens, handwriting tremor spirals) to detect which of the 4 diseases matches the visual pattern.
- **100% Software-Based (No External Hardware):**
  - Runs purely in the browser.
  - Acoustic screening uses standard built-in laptop/smartphone microphones or file uploads.
  - Vitals and laboratory parameters are entered from routine clinical checkup or lab reports.
- **Unified Single Web Platform:** Built with Streamlit, Scikit-Learn, and Librosa with persistent session state and consolidated patient health passport export.

---

## 📊 Dataset Sources & Clinical Provenance

All datasets used in this system are validated international academic and clinical benchmarks:

| Disease Module | Benchmark Dataset Source | Sample Count | Key Features Analyzed | Machine Learning Model |
| :--- | :--- | :--- | :--- | :--- |
| **Parkinson's Disease** | Acoustic Phonation Corpus & Voice Telemonitoring Benchmark | Audio phonation samples | 20 MFCCs, Zero-Crossing Rate, RMS Energy, Spectral Centroid (23 total) | Random Forest Classifier (300 estimators) |
| **Heart Disease** | **UCI Cleveland Clinic Foundation** *(Detrano et al.)* | 303 patient records | Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak ST depression, ST Slope, Vessels, Thal | Random Forest Classifier (83.6% Accuracy, 0.916 ROC-AUC) |
| **Diabetes Mellitus** | **NIDDK Pima Indians Diabetes Dataset** *(National Institutes of Health)* | 768 patient records | Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age | Random Forest Classifier (73.4% Accuracy, 0.808 ROC-AUC) |
| **Chronic Kidney Disease** | **UCI Apollo Hospitals Dataset** *(Dr. P. Soundarapandian et al.)* | 400 patient records | Age, Blood Pressure, Specific Gravity, Albumin, Sugar, Blood Glucose Random, Blood Urea, Serum Creatinine, Hemoglobin, Hypertension, Diabetes | Random Forest Classifier (98.8% 5-Fold Cross-Validation) |

---

## 📁 Repository Organization

```
D - Project/
├── app.py                     # Single Unified Multi-Disease Web Application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & viva guide
├── datasets/                  # Benchmark clinical datasets
│   ├── heart_disease.csv      # UCI Cleveland Heart Disease dataset
│   ├── diabetes.csv           # NIDDK Pima Indians Diabetes dataset
│   └── kidney_disease.csv     # UCI Apollo Hospitals CKD dataset
├── models/                    # Trained models & feature scalers
│   ├── audio_parkinsons_model.pkl
│   ├── audio_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_features.pkl
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_features.pkl
│   ├── kidney_model.pkl
│   ├── kidney_scaler.pkl
│   └── kidney_features.pkl
├── scripts/                   # Reproducible model training scripts
│   ├── train_heart_model.py
│   ├── train_diabetes_model.py
│   └── train_kidney_model.py
└── Parkinson2.1-main/         # Original Parkinson's codebase & audio assets
```

---

## 🚀 Running the Project Locally

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed.

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the directory
cd "D - Project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Single Unified Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 1-Click Free Cloud Deployment Guide

Because the system uses `librosa` for audio frequency extraction, **Streamlit Community Cloud** (`share.streamlit.io`) is the ideal deployment platform:

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Intelligent Healthcare Monitoring Multi-Disease System"
   git remote add origin https://github.com/YOUR_USERNAME/intelligent-healthcare-system.git
   git push -u origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New App"**:
   - Repository: `YOUR_USERNAME/intelligent-healthcare-system`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy!**
5. You will receive a permanent, live, shareable URL (e.g. `https://intelligent-healthcare-system.streamlit.app`).

---

