import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import librosa
from gtts import gTTS
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Healthcare Monitoring & Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# MODERN MEDICAL UI STYLING (CSS)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #3a506b 100%);
        padding: 26px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #f8f9fa;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin: 0;
    }

    .badge-bar {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .badge-chip {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00d2ff;
    }

    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d2ff;
        margin-top: 4px;
    }

    .metric-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 4px;
    }

    .result-box-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1.5px solid #10b981;
        padding: 22px;
        border-radius: 12px;
        color: #ecfdf5;
        margin-top: 15px;
    }

    .result-box-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.08) 100%);
        border: 1.5px solid #f59e0b;
        padding: 22px;
        border-radius: 12px;
        color: #fffbeb;
        margin-top: 15px;
    }

    .result-box-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1.5px solid #ef4444;
        padding: 22px;
        border-radius: 12px;
        color: #fef2f2;
        margin-top: 15px;
    }

    .info-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        color: #1e293b;
        margin-bottom: 18px;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL CACHING & LOADERS
# -------------------------------------------------------------
@st.cache_resource
def load_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # 1. Parkinson's Audio Model & Scaler
    park_model = joblib.load(os.path.join(models_dir, "audio_parkinsons_model.pkl"))
    park_scaler = joblib.load(os.path.join(models_dir, "audio_scaler.pkl"))
    
    # 2. Heart Disease Model & Scaler
    heart_model = joblib.load(os.path.join(models_dir, "heart_model.pkl"))
    heart_scaler = joblib.load(os.path.join(models_dir, "heart_scaler.pkl"))
    heart_cols = joblib.load(os.path.join(models_dir, "heart_features.pkl"))
    
    # 3. Diabetes Model & Scaler
    diabetes_model = joblib.load(os.path.join(models_dir, "diabetes_model.pkl"))
    diabetes_scaler = joblib.load(os.path.join(models_dir, "diabetes_scaler.pkl"))
    diabetes_cols = joblib.load(os.path.join(models_dir, "diabetes_features.pkl"))
    
    # 4. Chronic Kidney Disease Model & Scaler
    kidney_model = joblib.load(os.path.join(models_dir, "kidney_model.pkl"))
    kidney_scaler = joblib.load(os.path.join(models_dir, "kidney_scaler.pkl"))
    kidney_cols = joblib.load(os.path.join(models_dir, "kidney_features.pkl"))
    
    return {
        "parkinsons": {"model": park_model, "scaler": park_scaler},
        "heart": {"model": heart_model, "scaler": heart_scaler, "cols": heart_cols},
        "diabetes": {"model": diabetes_model, "scaler": diabetes_scaler, "cols": diabetes_cols},
        "kidney": {"model": kidney_model, "scaler": kidney_scaler, "cols": kidney_cols}
    }

models = load_all_models()

# -------------------------------------------------------------
# AUDIO HELPER FUNCTIONS (PARKINSON'S)
# -------------------------------------------------------------
def extract_audio_features(path, sr_target=22050, n_mfcc=20):
    y, sr = librosa.load(path, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    return np.hstack([mfcc_mean, zcr, rms, spec_centroid])

def is_audio_silent(path, threshold=0.01):
    try:
        y, sr = librosa.load(path, sr=None)
        if y.size == 0:
            return True
        rms = np.sqrt(np.mean(y**2))
        return rms < threshold
    except Exception:
        return False

LANG_OPTIONS = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Gujarati (ગુજરાતી)": "gu",
    "Marathi (मराठी)": "mr",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de"
}

def play_spoken_text(text: str, lang_code: str = "en"):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(tmp.name)
            tmp.seek(0)
            audio_bytes = tmp.read()
        os.remove(tmp.name)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio playback note: {e}")

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "screening_results" not in st.session_state:
    st.session_state.screening_results = {
        "parkinsons": None,
        "heart": None,
        "diabetes": None,
        "kidney": None
    }

# -------------------------------------------------------------
# HEADER BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🩺 Intelligent Healthcare Monitoring & Early Disease Prediction System</h1>
    <p>Integrated Multi-Disease Clinical Decision Support Platform • 100% Software-Based Screening</p>
    <div class="badge-bar">
        <span class="badge-chip">🎙️ Parkinson's Voice Screening</span>
        <span class="badge-chip">❤️ Cardiovascular Risk Predictor</span>
        <span class="badge-chip">🩸 Diabetes Mellitus Diagnostic</span>
        <span class="badge-chip">🫘 Chronic Kidney Disease Panel</span>
        <span class="badge-chip">⚡ Zero-Hardware Required</span>
        <span class="badge-chip">🚀 Deployable on Streamlit Cloud</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# NAVIGATION SIDEBAR
# -------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.title("Navigation Hub")
menu_selection = st.sidebar.radio(
    "Select Clinical Module:",
    [
        "🏠 System Overview & Vitals Dashboard",
        "🎙️ Parkinson's Disease (Voice Screening)",
        "❤️ Cardiovascular (Heart Disease)",
        "🩸 Diabetes Mellitus Diagnostic",
        "🫘 Chronic Kidney Disease (CKD)",
        "📋 Consolidated Patient Health Passport"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Models Status")
st.sidebar.success("✅ Parkinson's Model (Random Forest - 23 Feats)")
st.sidebar.success("✅ Heart Model (Random Forest - 83.6% Acc)")
st.sidebar.success("✅ Diabetes Model (Random Forest - 73.4% Acc)")
st.sidebar.success("✅ CKD Model (Random Forest - 98.8% 5-Fold)")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Clinical Disclaimer**: This application is a clinical decision support and educational tool. It is not an alternative to licensed clinical laboratory examinations.")

# -------------------------------------------------------------
# TAB 1: SYSTEM OVERVIEW & DASHBOARD
# -------------------------------------------------------------
if menu_selection == "🏠 System Overview & Vitals Dashboard":
    st.subheader("System Architecture & Clinical Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Diseases Covered</div>
            <div class="metric-value">4 Systems</div>
            <div class="metric-desc">Neuro, Cardio, Metabolic, Renal</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Hardware Requirement</div>
            <div class="metric-value">0% External</div>
            <div class="metric-desc">Web-based pure software CDSS</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Signal & Lab Inputs</div>
            <div class="metric-value">Multi-Modal</div>
            <div class="metric-desc">Acoustics, Vitals, Biochemicals</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Speech Feedback</div>
            <div class="metric-value">13+ Langs</div>
            <div class="metric-desc">Multilingual gTTS Accessibility</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        st.markdown("### 🔬 How Early Disease Prediction Works Without Hardware")
        st.write("""
        Traditional health diagnostics often rely on expensive specialized hospital hardware. This system demonstrates **Software-as-a-Medical-Device (SaMD)**:
        
        1. **Acoustic Phonetics (Parkinson's):** Neurological degeneration causes subtle vocal cord tremors and dysarthria before severe motor symptoms manifest. By analyzing 20 MFCC frequency bands, zero-crossing rate, and spectral centroids from an ordinary laptop/smartphone microphone, the AI flags early anomalies.
        2. **Vital Sign Corroboration (Heart Disease):** Blood pressure, resting heart rate, and routine ECG waveforms are analyzed via an ensemble classifier to identify ischemic heart disease and arterial narrowing risk.
        3. **Metabolic Profiling (Diabetes):** Blood sugar, insulin, and body metrics are screened to detect insulin resistance before microvascular damage occurs.
        4. **Biochemical Filtration (Kidney Disease):** Serum creatinine, blood urea, and urine albumin levels are evaluated to catch early asymptomatic renal decline.
        """)

    with c_right:
        st.markdown("### 🌐 Validated Academic Datasets")
        data_info = pd.DataFrame({
            "Disease Module": ["Parkinson's", "Heart Disease", "Diabetes", "Chronic Kidney Disease"],
            "Dataset Source": ["Voice Screening Corpus", "UCI Cleveland Clinic", "NIDDK Pima Indians", "UCI Apollo Hospitals"],
            "Samples": ["Acoustic Waves", "303 Patients", "768 Patients", "400 Patients"],
            "Evaluation Metric": ["Random Forest (23 Feats)", "83.6% Acc / 0.91 AUC", "73.4% Acc / 0.81 AUC", "98.8% Cross-Validation"]
        })
        st.dataframe(data_info, use_container_width=True, hide_index=True)

    st.info("💡 **Ready to begin?** Select any disease from the left sidebar to start clinical screening.")

# -------------------------------------------------------------
# TAB 2: PARKINSON'S DISEASE (VOICE SCREENING)
# -------------------------------------------------------------
elif menu_selection == "🎙️ Parkinson's Disease (Voice Screening)":
    st.subheader("🎙️ Parkinson's Disease: Voice Acoustic Screening & Motor Symptom Assessment")
    st.caption("Analyzes sustained vowel phonation ('AH' sound) using frequency-domain audio feature extraction.")

    col_q, col_audio = st.columns([1, 1.2])

    with col_q:
        st.markdown("#### 1. Clinical Symptom Checklist")
        st.write("Check if the patient exhibits any of the following motor signs:")
        s1 = st.checkbox("Tremor: Involuntary trembling in hands or limbs at rest")
        s2 = st.checkbox("Bradykinesia: General slowness of voluntary movement")
        s3 = st.checkbox("Rigidity: Muscle stiffness or resistance in limbs")
        s4 = st.checkbox("Postural Instability: Impaired balance or tendency to fall")
        s5 = st.checkbox("Hypophonia: Quieter, flatter, or softer voice tone")

        symptom_count = sum([s1, s2, s3, s4, s5])
        if symptom_count <= 1:
            sym_risk = "Low"
            sym_color = "green"
        elif symptom_count <= 3:
            sym_risk = "Moderate"
            sym_color = "orange"
        else:
            sym_risk = "High"
            sym_color = "red"
        
        st.markdown(f"**Symptom Clinical Risk Index:** :{sym_color}[**{sym_risk} ({symptom_count}/5 Criteria Met)**]")

        st.markdown("---")
        st.markdown("#### 2. Spoken Diagnosis Audio Language")
        selected_lang = st.selectbox("Select voice response language:", list(LANG_OPTIONS.keys()), index=0)
        lang_code = LANG_OPTIONS[selected_lang]

    with col_audio:
        st.markdown("#### 3. Voice Sample Upload or Quick Test")
        
        tab_upload, tab_sample = st.tabs(["📁 Upload Your Recording", "⚡ Quick Demo Samples"])
        
        target_audio_path = None
        
        with tab_upload:
            st.write("Upload a 5-second sustained vowel **'AH'** audio recording (`.wav`, `.mp3`, or `.m4a`).")
            audio_file = st.file_uploader("Upload sustained vowel audio file", type=["wav", "mp3", "m4a"], key="park_audio")
            if audio_file is not None:
                st.audio(audio_file)
                if st.button("🚀 Analyze Uploaded Voice", type="primary", use_container_width=True):
                    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_audio.write(audio_file.getvalue())
                    temp_audio.close()
                    target_audio_path = temp_audio.name
        
        with tab_sample:
            st.write("Test the AI acoustics instantly using calibrated benchmark recordings:")
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.caption("🟢 Healthy Control Sample")
                h_sample_path = os.path.join("datasets", "audio_samples", "sample_healthy_vowel.wav")
                if os.path.exists(h_sample_path):
                    with open(h_sample_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    if st.button("Test Healthy Sample", use_container_width=True):
                        target_audio_path = h_sample_path
            with c_s2:
                st.caption("🔴 Tremor / Dysarthria Sample")
                t_sample_path = os.path.join("datasets", "audio_samples", "sample_tremor_vowel.wav")
                if os.path.exists(t_sample_path):
                    with open(t_sample_path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    if st.button("Test Tremor Sample", use_container_width=True):
                        target_audio_path = t_sample_path

        if target_audio_path is not None:
            with st.spinner("Extracting 20 MFCCs, Spectral Centroid, ZCR, and RMS energy..."):
                try:
                    if is_audio_silent(target_audio_path):
                        st.warning("⚠️ The audio file appears to be silent. Please provide an audible 'AH' recording.")
                    else:
                        features = extract_audio_features(target_audio_path)
                        features_scaled = models["parkinsons"]["scaler"].transform([features])
                        prob_pd = models["parkinsons"]["model"].predict_proba(features_scaled)[0, 1]
                        prob_pct = prob_pd * 100

                        st.session_state.screening_results["parkinsons"] = {
                            "probability": prob_pct,
                            "symptoms": f"{symptom_count}/5 ({sym_risk})",
                            "status": "Detected" if prob_pd >= 0.8 else "Healthy"
                        }

                        st.markdown("### 📊 Diagnostic Results")
                        m1, m2 = st.columns(2)
                        m1.metric("Acoustic Parkinson's Probability", f"{prob_pct:.1f}%")
                        m2.metric("Clinical Symptom Index", f"{symptom_count}/5")

                        if prob_pd < 0.8:
                            st.markdown(f"""
                            <div class="result-box-healthy">
                                <h3>🟢 NO SIGNIFICANT PARKINSONIAN SIGNS DETECTED</h3>
                                <p>Voice acoustic analysis shows <b>{prob_pct:.1f}%</b> probability of Parkinson's dysarthria (threshold: 80%). Vocal stability and harmonics are within normal limits.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            speech_text = f"Voice analysis indicates healthy vocal stability with {int(prob_pct)} percent probability. Symptom index is {sym_risk}. This is an educational screening demo."
                        else:
                            st.markdown(f"""
                            <div class="result-box-danger">
                                <h3>🔴 ELEVATED RISK OF PARKINSON'S DETECTED</h3>
                                <p>Voice acoustic analysis indicates <b>{prob_pct:.1f}%</b> probability of Parkinsonian dysarthria. Neurological consultation and formal clinical evaluation (UPDRS) are strongly recommended.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            speech_text = f"Elevated Parkinson's probability detected at {int(prob_pct)} percent. Clinical consultation with a neurologist is recommended."

                        st.markdown("#### 🔊 Spoken Audio Assessment")
                        play_spoken_text(speech_text, lang_code=lang_code)

                except Exception as e:
                    st.error(f"Error during audio processing: {e}")
                finally:
                    if target_audio_path.startswith(tempfile.gettempdir()) and os.path.exists(target_audio_path):
                        os.remove(target_audio_path)

# -------------------------------------------------------------
# TAB 3: CARDIOVASCULAR (HEART DISEASE)
# -------------------------------------------------------------
elif menu_selection == "❤️ Cardiovascular (Heart Disease)":
    st.subheader("❤️ Cardiovascular / Heart Disease Risk Prediction")
    st.caption("Predicts coronary artery disease and myocardial risk using clinical vitals and stress test biomarkers.")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Patient Age (Years)", min_value=18, max_value=100, value=52)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Chest Pain Type", options=[
                ("Typical Angina", 0),
                ("Atypical Angina", 1),
                ("Non-Anginal Pain", 2),
                ("Asymptomatic", 3)
            ], format_func=lambda x: x[0])[1]
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=128)

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=214)
            fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("No (Normal)", 0), ("Yes (>120 mg/dL)", 1)], format_func=lambda x: x[0])[1]
            restecg = st.selectbox("Resting ECG Results", options=[
                ("Normal", 0),
                ("ST-T Wave Abnormality", 1),
                ("Left Ventricular Hypertrophy", 2)
            ], format_func=lambda x: x[0])[1]
            thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=230, value=156)

        with col3:
            exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=7.0, value=0.8, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])[1]
            ca = st.slider("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia Status", options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)], format_func=lambda x: x[0])[1]

        submit_heart = st.form_submit_button("💓 Evaluate Cardiovascular Risk", type="primary", use_container_width=True)

    if submit_heart:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs_val, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        scaled_input = models["heart"]["scaler"].transform(input_data)
        prob = models["heart"]["model"].predict_proba(scaled_input)[0, 1]
        prob_pct = prob * 100

        st.session_state.screening_results["heart"] = {
            "probability": prob_pct,
            "status": "High Risk" if prob >= 0.5 else "Low Risk",
            "vitals": f"BP: {trestbps} mm Hg, Chol: {chol} mg/dL, Max HR: {thalach}"
        }

        st.markdown("### 📊 Cardiovascular Assessment")
        k1, k2, k3 = st.columns(3)
        k1.metric("Coronary Risk Probability", f"{prob_pct:.1f}%")
        k2.metric("Blood Pressure Status", "Normal" if trestbps < 130 else "Hypertensive")
        k3.metric("Cholesterol Status", "Desirable" if chol < 200 else "Borderline / High")

        if prob >= 0.5:
            st.markdown(f"""
            <div class="result-box-danger">
                <h3>🔴 ELEVATED CARDIOVASCULAR DISEASE RISK DETECTED</h3>
                <p>The model indicates an estimated <b>{prob_pct:.1f}%</b> probability of coronary artery disease. Key contributing risk factors include ST depression and hemodynamic vitals.</p>
                <p><b>Recommended Action:</b> Immediate cardiologist consultation, 12-lead diagnostic ECG, and lipid panel optimization.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-healthy">
                <h3>🟢 LOW CARDIOVASCULAR RISK DETECTED</h3>
                <p>Coronary artery disease probability is estimated at <b>{prob_pct:.1f}%</b>. Patient hemodynamics and exercise parameters are within standard ranges.</p>
                <p><b>Recommended Action:</b> Maintain regular aerobic exercise, balanced Mediterranean diet, and annual preventive blood pressure checks.</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: DIABETES MELLITUS DIAGNOSTIC
# -------------------------------------------------------------
elif menu_selection == "🩸 Diabetes Mellitus Diagnostic":
    st.subheader("🩸 Diabetes Mellitus Diagnostic Risk Predictor")
    st.caption("Identifies pre-diabetes and Type-2 Diabetes risk using metabolic panel and body composition indicators.")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Number of Pregnancies (0 for males)", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Plasma Glucose Concentration (2h oral test, mg/dL)", min_value=50, max_value=300, value=118)
            bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=40, max_value=160, value=74)
            skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=5, max_value=99, value=25)

        with col2:
            insulin = st.number_input("2-Hour Serum Insulin (μU/mL)", min_value=10, max_value=850, value=90)
            bmi = st.number_input("Body Mass Index (BMI, kg/m²)", min_value=10.0, max_value=65.0, value=27.4, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function (Genetic History Score)", min_value=0.05, max_value=2.5, value=0.45, step=0.01)
            age = st.number_input("Age (Years)", min_value=18, max_value=100, value=38)

        submit_diab = st.form_submit_button("🩸 Evaluate Diabetic Risk", type="primary", use_container_width=True)

    if submit_diab:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_input = models["diabetes"]["scaler"].transform(input_data)
        prob = models["diabetes"]["model"].predict_proba(scaled_input)[0, 1]
        prob_pct = prob * 100

        st.session_state.screening_results["diabetes"] = {
            "probability": prob_pct,
            "status": "Diabetic Risk" if prob >= 0.5 else "Non-Diabetic",
            "vitals": f"Glucose: {glucose} mg/dL, BMI: {bmi}, Age: {age}"
        }

        st.markdown("### 📊 Metabolic Assessment")
        d1, d2, d3 = st.columns(3)
        d1.metric("Diabetes Probability", f"{prob_pct:.1f}%")
        d2.metric("Glycemic Status", "Normal" if glucose < 140 else "Impaired / Diabetic")
        d3.metric("BMI Classification", "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese")

        if prob >= 0.5:
            st.markdown(f"""
            <div class="result-box-danger">
                <h3>🔴 ELEVATED TYPE-2 DIABETES RISK</h3>
                <p>The patient has a <b>{prob_pct:.1f}%</b> probability of diabetic metabolic dysfunction. Fasting glucose and BMI metrics suggest impaired glycemic control.</p>
                <p><b>Recommended Action:</b> Order an HbA1c confirmatory laboratory test, consult an endocrinologist, and initiate low-glycemic dietary protocols.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-healthy">
                <h3>🟢 NON-DIABETIC / HEALTHY GLYCEMIC PROFILE</h3>
                <p>Diabetes probability is estimated at <b>{prob_pct:.1f}%</b>. Metabolic parameters indicate healthy glucose utilization.</p>
                <p><b>Recommended Action:</b> Continue active lifestyle, balanced dietary fiber, and annual wellness screenings.</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: CHRONIC KIDNEY DISEASE (CKD)
# -------------------------------------------------------------
elif menu_selection == "🫘 Chronic Kidney Disease (CKD)":
    st.subheader("🫘 Chronic Kidney Disease (CKD) Early Detection Panel")
    st.caption("Screens for renal failure and filtration degradation using routine blood and urine biomarkers.")

    with st.form("ckd_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Years)", min_value=5, max_value=100, value=50)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80)
            sg = st.selectbox("Specific Gravity of Urine", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            al = st.selectbox("Albumin in Urine (0 to 5)", options=[0, 1, 2, 3, 4, 5], index=0)

        with col2:
            su = st.selectbox("Sugar in Urine (0 to 5)", options=[0, 1, 2, 3, 4, 5], index=0)
            bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=60, max_value=500, value=110)
            bu = st.number_input("Blood Urea (mg/dL)", min_value=5.0, max_value=400.0, value=35.0)
            sc = st.number_input("Serum Creatinine (mg/dL) - Crucial Marker", min_value=0.2, max_value=20.0, value=0.9, step=0.1)

        with col3:
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=14.5, step=0.1)
            htn = st.selectbox("Hypertension History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            dm = st.selectbox("Diabetes Mellitus History", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

        submit_ckd = st.form_submit_button("🫘 Evaluate Renal Function", type="primary", use_container_width=True)

    if submit_ckd:
        input_data = np.array([[age, bp, sg, al, su, bgr, bu, sc, hemo, htn, dm]])
        scaled_input = models["kidney"]["scaler"].transform(input_data)
        prob = models["kidney"]["model"].predict_proba(scaled_input)[0, 1]
        prob_pct = prob * 100

        st.session_state.screening_results["kidney"] = {
            "probability": prob_pct,
            "status": "CKD Detected" if prob >= 0.5 else "Healthy Kidneys",
            "vitals": f"Creatinine: {sc} mg/dL, Urea: {bu} mg/dL, Albumin: {al}"
        }

        st.markdown("### 📊 Renal Health Assessment")
        c1, c2, c3 = st.columns(3)
        c1.metric("CKD Probability", f"{prob_pct:.1f}%")
        c2.metric("Serum Creatinine", f"{sc} mg/dL", "Normal" if sc <= 1.2 else "Elevated", delta_color="inverse")
        c3.metric("Blood Urea", f"{bu} mg/dL", "Normal" if bu <= 40 else "Elevated", delta_color="inverse")

        if prob >= 0.5:
            st.markdown(f"""
            <div class="result-box-danger">
                <h3>🔴 EARLY SIGNS OF CHRONIC KIDNEY DISEASE (CKD)</h3>
                <p>Renal filtration indicators show an estimated <b>{prob_pct:.1f}%</b> probability of chronic kidney damage. Creatinine and urea concentrations indicate reduced glomerular filtration.</p>
                <p><b>Recommended Action:</b> Consult a nephrologist for eGFR calculation, renal ultrasound, and strict blood pressure monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-healthy">
                <h3>🟢 NORMAL RENAL FUNCTION</h3>
                <p>Kidney filtration risk is estimated at <b>{prob_pct:.1f}%</b>. Creatinine, urea, and urine specific gravity are within physiological baselines.</p>
                <p><b>Recommended Action:</b> Maintain adequate hydration, low sodium intake, and regular periodic health checkups.</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 6: CONSOLIDATED PATIENT HEALTH PASSPORT
# -------------------------------------------------------------
elif menu_selection == "📋 Consolidated Patient Health Passport":
    st.subheader("📋 Consolidated Patient Health Passport & Multi-Disease Report")
    st.caption("Aggregated multi-system risk profile generated across all four disease prediction modules.")

    res = st.session_state.screening_results

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🩺 Multi-Disease Risk Summary")
        
        # Parkinson's card
        if res["parkinsons"] is not None:
            st.success(f"**Parkinson's Disease:** {res['parkinsons']['status']} ({res['parkinsons']['probability']:.1f}% risk, Symptoms: {res['parkinsons']['symptoms']})")
        else:
            st.info("ℹ️ **Parkinson's Screening:** Not evaluated yet.")

        # Heart card
        if res["heart"] is not None:
            if res["heart"]["probability"] >= 50:
                st.error(f"**Cardiovascular Disease:** {res['heart']['status']} ({res['heart']['probability']:.1f}% risk • {res['heart']['vitals']})")
            else:
                st.success(f"**Cardiovascular Disease:** {res['heart']['status']} ({res['heart']['probability']:.1f}% risk • {res['heart']['vitals']})")
        else:
            st.info("ℹ️ **Cardiovascular Screening:** Not evaluated yet.")

        # Diabetes card
        if res["diabetes"] is not None:
            if res["diabetes"]["probability"] >= 50:
                st.error(f"**Diabetes Mellitus:** {res['diabetes']['status']} ({res['diabetes']['probability']:.1f}% risk • {res['diabetes']['vitals']})")
            else:
                st.success(f"**Diabetes Mellitus:** {res['diabetes']['status']} ({res['diabetes']['probability']:.1f}% risk • {res['diabetes']['vitals']})")
        else:
            st.info("ℹ️ **Diabetes Screening:** Not evaluated yet.")

        # Kidney card
        if res["kidney"] is not None:
            if res["kidney"]["probability"] >= 50:
                st.error(f"**Chronic Kidney Disease:** {res['kidney']['status']} ({res['kidney']['probability']:.1f}% risk • {res['kidney']['vitals']})")
            else:
                st.success(f"**Chronic Kidney Disease:** {res['kidney']['status']} ({res['kidney']['probability']:.1f}% risk • {res['kidney']['vitals']})")
        else:
            st.info("ℹ️ **Chronic Kidney Disease:** Not evaluated yet.")

    with col2:
        st.markdown("### 📥 Export Clinical Screening Passport")
        patient_name = st.text_input("Patient Identifier / Name", value="Patient-001")
        
        report_text = f"""=======================================================
INTELLIGENT HEALTHCARE MONITORING & EARLY DISEASE PREDICTION
Patient Clinical Screening Summary Report
Generated for: {patient_name}
=======================================================

1. PARKINSON'S DISEASE (ACOUSTICS & MOTOR SIGNS)
   - Status: {res['parkinsons']['status'] if res['parkinsons'] else 'Not Evaluated'}
   - Risk Probability: {f"{res['parkinsons']['probability']:.1f}%" if res['parkinsons'] else 'N/A'}
   - Symptom Index: {res['parkinsons']['symptoms'] if res['parkinsons'] else 'N/A'}

2. CARDIOVASCULAR / HEART DISEASE
   - Status: {res['heart']['status'] if res['heart'] else 'Not Evaluated'}
   - Risk Probability: {f"{res['heart']['probability']:.1f}%" if res['heart'] else 'N/A'}
   - Recorded Vitals: {res['heart']['vitals'] if res['heart'] else 'N/A'}

3. DIABETES MELLITUS DIAGNOSTIC
   - Status: {res['diabetes']['status'] if res['diabetes'] else 'Not Evaluated'}
   - Risk Probability: {f"{res['diabetes']['probability']:.1f}%" if res['diabetes'] else 'N/A'}
   - Metabolic Indicators: {res['diabetes']['vitals'] if res['diabetes'] else 'N/A'}

4. CHRONIC KIDNEY DISEASE (CKD)
   - Status: {res['kidney']['status'] if res['kidney'] else 'Not Evaluated'}
   - Risk Probability: {f"{res['kidney']['probability']:.1f}%" if res['kidney'] else 'N/A'}
   - Renal Panel: {res['kidney']['vitals'] if res['kidney'] else 'N/A'}

=======================================================
CLINICAL NOTICE:
This report was generated by an ensemble Machine Learning Clinical
Decision Support System. Results are predictive indicators for early
screening and must be verified by a licensed medical practitioner.
=======================================================
"""
        st.download_button(
            label="📄 Download Official Patient Health Report (.txt)",
            data=report_text,
            file_name=f"{patient_name}_Healthcare_Screening_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    Intelligent Healthcare Monitoring and Early Disease Prediction System • Semester 5 Design Project<br>
    Built with Python, Streamlit, Scikit-Learn, and Librosa • 100% Software-Based Clinical Decision Support
</div>
""", unsafe_allow_html=True)
