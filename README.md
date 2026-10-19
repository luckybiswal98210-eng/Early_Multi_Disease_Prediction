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

---

## 🌐 Full Production Deployment Architecture (Vercel + Next.js + MongoDB + Render)

This project features a complete **Production Web Application** with:
- **Frontend:** Next.js (React 19) deployable on **Vercel**
- **Database:** **MongoDB Atlas** for multi-user self-registration, authentication, and lifetime diagnostic logs
- **ML Backend:** Python FastAPI microservice deployable on **Render** / **Railway**
- **Admin Portal:** Exclusive administrative dashboard for monitoring all registered users and screening records

```
┌─────────────────────────────────────────────────────────────────┐
│               Vercel Frontend (Next.js App)                     │
│  - Self-Registration & Auth (Email / Password)                 │
│  - Patient Portal (4 Diseases, Vitals, Audio Recording)         │
│  - 👑 Master Admin Portal (All Users, All Records, CSV Export) │
└───────────────────────┬───────────────────────┬─────────────────┘
                        │                       │
                        ▼                       ▼
            ┌──────────────────────┐  ┌──────────────────────┐
            │  MongoDB Atlas Cloud │  │ FastAPI ML Backend   │
            │  - Users Collection  │  │ - Scikit-Learn Models│
            │  - Screening Logs    │  │ - Librosa 20 MFCCs   │
            │  (Free Cloud Tier)   │  │ (Render / Railway)   │
            └──────────────────────┘  └──────────────────────┘
```

### 1. Deploy Frontend to Vercel (Free)
1. Go to **[vercel.com](https://vercel.com)** and log in with GitHub.
2. Click **"Add New..."** -> **"Project"** and import `luckybiswal98210-eng/Early_Multi_Disease_Prediction`.
3. In **Project Settings**:
   - Set **Root Directory** to `frontend`
4. Under **Environment Variables**, add:
   - `MONGODB_URI`: Your free MongoDB Atlas connection string (`mongodb+srv://...`)
   - `NEXT_PUBLIC_API_URL`: URL of your deployed FastAPI backend (e.g., `https://healthcare-ml-backend.onrender.com`)
5. Click **Deploy!** You will get a live URL (e.g. `https://early-multi-disease-prediction.vercel.app`).

### 2. Set Up MongoDB Atlas (Free Cloud Database)
1. Sign up at **[mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)**.
2. Create a **Free Shared Cluster (M0)**.
3. Under **Database Access**, create a user (username + password).
4. Under **Network Access**, allow IP `0.0.0.0/0` (access from anywhere).
5. Click **Connect** -> **Drivers** and copy your `mongodb+srv://...` URI.
6. Paste it into your Vercel Environment Variables.

### 3. Deploy ML Backend to Render (Free)
1. Sign up at **[render.com](https://render.com)**.
2. Click **"New +"** -> **"Web Service"** and connect this repository.
3. Settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**. Copy the generated URL into Vercel's `NEXT_PUBLIC_API_URL`.

---

## 👥 Multi-User Roles & Admin Credentials

- **Self-Service Registration:** Any user visiting the website can sign up with their Full Name, Email, Password, Age, and Gender. Their test results and health passports are private to their account.
- **👑 Administrator Portal:**
  - **Admin Email:** `admin@health.org`
  - **Admin Password:** `admin123`
  - When logged in as Admin, a golden **👑 Master Admin Portal** tab unlocks with:
    - Registered Users Directory (Name, Email, Age, Join Date)
    - Master Screening Logs across all patients
    - One-Click **Export All Records to CSV**

---

## 💻 Alternative: Running Streamlit App Locally
If you prefer running the single-file Streamlit version:
```bash
pip install -r requirements.txt
streamlit run app.py
```

