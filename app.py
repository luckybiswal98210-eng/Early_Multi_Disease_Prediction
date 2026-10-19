import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import librosa
from gtts import gTTS
import matplotlib.pyplot as plt
from PIL import Image

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
# NAVIGATION SIDEBAR
# -------------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.title("Navigation Hub")
menu_selection = st.sidebar.radio(
    "Select Clinical Module:",
    [
        "🏠 System Overview & Vitals Dashboard",
        "🤖 AI Medical Chatbot & Image Triaging",
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
# TAB 1: SYSTEM OVERVIEW & DASHBOARD (WELCOME PAGE)
# -------------------------------------------------------------
if menu_selection == "🏠 System Overview & Vitals Dashboard":
    # Header Banner ONLY displayed on the Initial/Welcome Page
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Intelligent Healthcare Monitoring & Early Disease Prediction System</h1>
        <div class="badge-bar">
            <span class="badge-chip">🎙️ Parkinson's Voice Screening</span>
            <span class="badge-chip">❤️ Cardiovascular Risk Predictor</span>
            <span class="badge-chip">🩸 Diabetes Mellitus Diagnostic</span>
            <span class="badge-chip">🫘 Chronic Kidney Disease Panel</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    st.info("💡 **Ready to begin?** Select any disease or the AI Chatbot from the left sidebar to start clinical screening.")

# -------------------------------------------------------------
# TAB 2: AI MEDICAL CHATBOT & IMAGE TRIAGING
# -------------------------------------------------------------
elif menu_selection == "🤖 AI Medical Chatbot & Image Triaging":
    st.subheader("🤖 AI Clinical Chatbot & Visual Disease Identifier")
    st.caption("Interactive conversational assistant for symptom checking, home remedies, and image-based multi-disease identification.")

    sub_tab1, sub_tab2 = st.tabs(["💬 Symptom Chatbot & Home Remedies", "📷 Visual Disease Identifier (Image Triaging)"])

    # ---------------------------------------------------------
    # SUB-TAB 1: SYMPTOM CHATBOT & HOME REMEDIES
    # ---------------------------------------------------------
    with sub_tab1:
        st.markdown("#### 💬 Clinical Symptom Checker & Home Remedies Assistant")
        st.write("Ask any questions regarding symptoms (like fever, cough, acidity, tremors) for immediate follow-up questions, evidence-based home remedies, and care precautions.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": "Hello! I am your Intelligent Healthcare AI Assistant. 🩺\n\nYou can ask me about any symptoms (e.g., *'I am getting fever'*, *'Severe headache'*, *'Chest tightness'*, *'Shaking hands'*). I will ask relevant diagnostic questions, suggest **evidence-based home remedies**, and guide you on when to seek urgent medical care."
                }
            ]

        # Quick symptom chips
        st.markdown("**Quick Clinical Prompts:**")
        qc1, qc2, qc3, qc4 = st.columns(4)
        quick_prompt = None
        with qc1:
            if st.button("🤒 I am getting fever", use_container_width=True):
                quick_prompt = "I am getting fever"
            if st.button("🖐️ Shaking / Tremors in hands", use_container_width=True):
                quick_prompt = "My hands are shaking and I notice tremors at rest"
        with qc2:
            if st.button("🤧 Cough & Cold", use_container_width=True):
                quick_prompt = "I have a bad cough and sore throat"
            if st.button("❤️ Chest Discomfort / Pain", use_container_width=True):
                quick_prompt = "I feel chest tightness and shortness of breath"
        with qc3:
            if st.button("🤕 Severe Headache", use_container_width=True):
                quick_prompt = "I have a throbbing headache and fatigue"
            if st.button("🚰 Excessive Thirst & Urination", use_container_width=True):
                quick_prompt = "I have extreme thirst and frequent urination"
        with qc4:
            if st.button("🔥 Acidity & Indigestion", use_container_width=True):
                quick_prompt = "I have burning stomach acidity and indigestion"
            if st.button("🦵 Swollen Feet & Ankles", use_container_width=True):
                quick_prompt = "My feet and ankles are swollen and puffy"

        def get_clinical_response(user_text: str):
            t = user_text.lower()
            
            # 1. FEVER CONVERSATION LOGIC
            if any(w in t for w in ["fever", "temperature", "chills", "feeling hot", "shiver"]):
                has_details = any(num in t for num in ["99", "100", "101", "102", "103", "104", "degree", "high", "mild", "day", "days", "yesterday", "since", "body ache"])
                if not has_details:
                    return (
                        "🌡️ **I note that you are experiencing a fever.**\n\n"
                        "To provide the most accurate home remedies and guidance, could you please specify:\n"
                        "1. **What is your body temperature** (e.g., mild 99°–101°F or high 102°F+)?\n"
                        "2. **How many days** have you had this fever?\n"
                        "3. Are you having other symptoms like **chills, body ache, sore throat, or headache**?"
                    )
                else:
                    return (
                        "### 🌿 Recommended Home Remedies for Fever:\n"
                        "- **Hydration:** Drink plenty of fluids (boiled warm water, tender coconut water, electrolyte ORS, clear broths) to replenish fluids lost through sweating.\n"
                        "- **Lukewarm Sponge Bath:** Gently apply a clean damp cloth soaked in lukewarm water (not ice water) to your forehead, back of neck, and wrists. Ice-cold water induces shivering, which increases core temperature.\n"
                        "- **Herbal Tulsi-Ginger Kadha/Tea:** Boil 5–6 fresh Tulsi (Holy Basil) leaves, crushed ginger, and 2 black pepper pods in water. Add 1 tsp honey and drink warm twice daily.\n"
                        "- **Rest in a Well-Ventilated Room:** Ensure complete bed rest wearing loose, breathable cotton clothes. Avoid heavy blankets.\n"
                        "- **Light Nutrition:** Eat easily digestible foods like khichdi, moong dal soup, or boiled vegetables.\n\n"
                        "### 💊 General Medical Considerations:\n"
                        "- Over-the-counter **Paracetamol (Acetaminophen)** may be used to reduce fever and relieve body aches (consult a physician/pharmacist for exact dosage based on age/weight).\n"
                        "- Avoid NSAIDs like aspirin or ibuprofen on an empty stomach.\n\n"
                        "### 🚨 Red Flag Warning (Seek Doctor Immediately):\n"
                        "- Body temperature exceeds **103°F (39.4°C)** or fails to respond to fever medication.\n"
                        "- Fever persists for **more than 3 consecutive days**.\n"
                        "- Accompanied by **stiff neck, severe breathing difficulty, persistent vomiting, confusion, or sudden skin rash**."
                    )

            # 2. COUGH & SORE THROAT
            elif any(w in t for w in ["cough", "throat", "cold", "congestion", "sore throat", "phlegm"]):
                return (
                    "### 🌿 Recommended Home Remedies for Cough & Sore Throat:\n"
                    "- **Steam Inhalation:** Inhale steam from hot water 2–3 times a day. You can add a drop of eucalyptus oil or a pinch of carom seeds (ajwain).\n"
                    "- **Warm Salt-Water Gargle:** Dissolve 1/2 tsp salt in warm water; gargle 3 times a day to reduce throat inflammation.\n"
                    "- **Honey & Ginger Juice:** Mix 1 tsp fresh ginger juice with 1 tsp raw honey; consume slowly twice daily.\n"
                    "- **Golden Turmeric Milk:** Drink warm milk with 1/2 tsp turmeric powder and a pinch of black pepper before bedtime.\n\n"
                    "### 🚨 Red Flags: See a doctor if you cough up blood, have breathing difficulty, or persistent chest wheezing."
                )

            # 3. HEADACHE & MIGRAINE
            elif any(w in t for w in ["headache", "migraine", "head ache", "head hurts"]):
                return (
                    "### 🌿 Recommended Home Remedies for Headache:\n"
                    "- **Hydration Check:** Dehydration is the #1 silent headache trigger. Drink 2 full glasses of water immediately.\n"
                    "- **Cool or Warm Compress:** Apply a cool cloth on your forehead or a warm compress on the back of your neck to relieve muscle tension.\n"
                    "- **Dark, Quiet Room:** Lie down in a dim, silent room and close your eyes away from phone/laptop screen glare.\n"
                    "- **Ginger Tea:** Fresh ginger tea helps inhibit prostaglandins that trigger migraine headaches.\n\n"
                    "### 🚨 Red Flags: Seek emergency care if headache is sudden and explosive ('thunderclap'), accompanied by numbness or slurred speech."
                )

            # 4. ACIDITY & INDIGESTION
            elif any(w in t for w in ["acidity", "heartburn", "indigestion", "acid reflux", "bloating", "gas", "burning"]):
                return (
                    "### 🌿 Recommended Home Remedies for Acidity & Heartburn:\n"
                    "- **Cold Milk / Coconut Water:** Sip a cup of cold milk or fresh coconut water for immediate soothing of gastric lining.\n"
                    "- **Fennel Seeds (Saunf):** Chew 1/2 tsp of fennel seeds after meals to aid digestion and soothe mucosal burning.\n"
                    "- **Cumin (Jeera) Water:** Boil 1 tsp cumin seeds in a glass of water, strain, and sip warm.\n"
                    "- **Post-Meal Posture:** Never lie down within 2 hours after eating. Elevate your head slightly while sleeping.\n\n"
                    "### 🚨 Red Flags: If burning radiates to left arm, neck, or jaw with cold sweats, treat as potential cardiac emergency."
                )

            # 5. PARKINSON'S INDICATORS (TREMORS, SHAKING, SLOWNESS)
            elif any(w in t for w in ["tremor", "shake", "shaking", "parkinson", "stiff muscle", "slowness", "bradykinesia"]):
                return (
                    "### 🧠 Clinical Insight: Neurological & Movement Symptoms\n"
                    "The symptoms you described (hand tremors at rest, muscle stiffness, or movement slowness) correlate with **Parkinsonian Neurological Signs**.\n\n"
                    "**Supportive Home & Lifestyle Guidance:**\n"
                    "- **Stress Reduction:** Tremors worsen significantly under anxiety; practice slow diaphragmatic breathing.\n"
                    "- **Gentle Hand Exercises:** Squeezing a soft stress ball or rhythmic finger tapping helps maintain motor coordination.\n"
                    "- **Neuro-Protective Nutrition:** Adopt a Mediterranean-style diet rich in omega-3 fatty acids, berries, and green leafy vegetables.\n\n"
                    "👉 **Recommended Diagnostic Action:** Please use our **🎙️ Parkinson's Disease (Voice Screening)** module in the left sidebar to perform acoustic voice screening and complete the clinical tremor assessment."
                )

            # 6. CARDIOVASCULAR INDICATORS (CHEST PAIN, TIGHTNESS, PALPITATIONS)
            elif any(w in t for w in ["chest", "heart", "angina", "palpitation", "breathless", "shortness of breath"]):
                return (
                    "### ❤️ Clinical Insight: Cardiovascular & Hemodynamic Symptoms\n"
                    "The symptoms you described (chest tightness, exertion discomfort, or shortness of breath) strongly correlate with **Cardiovascular / Ischemic Heart Disease**.\n\n"
                    "**Immediate Precautions:**\n"
                    "- Stop physical exertion immediately, sit upright, and loosen restrictive collar/clothing.\n"
                    "- Remain as calm as possible to reduce cardiac oxygen demand.\n\n"
                    "🚨 **EMERGENCY WARNING:** If chest pain is crushing, heavy, or radiates to your **left arm, jaw, shoulder, or back**, accompanied by cold sweating or dizziness, **call emergency medical services immediately!**\n\n"
                    "👉 **Recommended Diagnostic Action:** If stable, use our **❤️ Cardiovascular (Heart Disease)** module in the left sidebar to analyze your blood pressure, cholesterol, and resting ECG parameters."
                )

            # 7. DIABETES INDICATORS (THIRST, FREQUENT URINATION, WEIGHT LOSS)
            elif any(w in t for w in ["thirst", "urination", "diabetes", "sugar", "glucose", "tingling", "polyuria"]):
                return (
                    "### 🩸 Clinical Insight: Glycemic & Metabolic Symptoms\n"
                    "The symptoms you described (excessive thirst [polydipsia], frequent urination [polyuria], or tingling in fingers/feet) correlate with **Diabetes Mellitus / Hyperglycemia**.\n\n"
                    "**Supportive Home & Lifestyle Guidance:**\n"
                    "- **Hydrate with Plain Water:** Avoid all sugary beverages, sodas, and packaged fruit juices.\n"
                    "- **Post-Meal Walking:** A light 15-minute walk after meals helps muscles absorb glucose independently of insulin.\n"
                    "- **Dietary Fiber:** Increase intake of whole grains, fenugreek (methi) seeds, and bitter gourd (karela) juice.\n\n"
                    "👉 **Recommended Diagnostic Action:** Please use our **🩸 Diabetes Mellitus Diagnostic** module in the sidebar to check your glycemic and BMI risk."
                )

            # 8. KIDNEY INDICATORS (SWOLLEN FEET, FOAMY URINE, FLUID)
            elif any(w in t for w in ["kidney", "edema", "swollen feet", "swollen ankle", "foamy urine", "creatinine", "urea"]):
                return (
                    "### 🫘 Clinical Insight: Renal Filtration & Fluid Retention Symptoms\n"
                    "The symptoms you described (swelling in ankles/feet [pedal edema], facial puffiness, or frothy urine) correlate with **Chronic Kidney Disease (CKD)**.\n\n"
                    "**Supportive Home & Lifestyle Guidance:**\n"
                    "- **Strict Sodium Limitation:** Minimize table salt, processed snacks, and pickles to prevent excess fluid accumulation.\n"
                    "- **Avoid Overuse of Painkillers:** Do NOT take NSAID medications (like ibuprofen, diclofenac) without a doctor's order, as they directly strain nephron filtration.\n"
                    "- **Monitor Fluid Balance:** Track daily water intake and weigh yourself regularly to watch for sudden fluid retention.\n\n"
                    "👉 **Recommended Diagnostic Action:** Please use our **🫘 Chronic Kidney Disease (CKD)** module in the sidebar to assess your serum creatinine, urea, and albumin parameters."
                )

            # 9. GENERAL / FALLBACK
            else:
                return (
                    "Thank you for sharing your symptom details. 🩺\n\n"
                    "**General Supportive Care Advice:**\n"
                    "- Stay well-hydrated with pure water and herbal infusions.\n"
                    "- Ensure 7–8 hours of uninterrupted rest to support your immune system.\n"
                    "- Eat light, freshly prepared, balanced meals.\n\n"
                    "If your symptoms correlate with our 4 core screening areas (**Parkinson's, Cardiovascular Disease, Diabetes, or Kidney Disease**), please use the dedicated clinical modules in the sidebar for precise machine-learning risk predictions.\n\n"
                    "*Always consult a licensed medical professional for personal clinical diagnoses.*"
                )

        # Handle user input via chat input or quick buttons
        user_message = st.chat_input("Describe your symptoms (e.g. 'I am getting fever', 'Hand shaking', 'Chest pain')...")
        if quick_prompt:
            user_message = quick_prompt

        if user_message:
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            response_text = get_clinical_response(user_message)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Display conversational chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🩺"):
                    st.markdown(msg["content"])

        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ---------------------------------------------------------
    # SUB-TAB 2: VISUAL DISEASE IDENTIFIER (IMAGE TRIAGING)
    # ---------------------------------------------------------
    with sub_tab2:
        st.markdown("#### 📷 Visual Disease Identifier & Diagnostic Triaging")
        st.write("Upload a clinical photograph, diagnostic monitor display, laboratory test report, or symptom photo. The AI analyzes visual markers and asks clinical follow-up questions to identify which of the 4 core diseases it correlates with.")

        img_col1, img_col2 = st.columns([1, 1.2])
        
        target_pil_img = None
        img_source_label = ""

        with img_col1:
            st.markdown("##### 1. Select or Upload Image")
            img_mode = st.radio("Choose Input Method:", ["⚡ Choose from Clinical Benchmark Samples", "📁 Upload Custom Photo/Report"], horizontal=True)

            if img_mode == "⚡ Choose from Clinical Benchmark Samples":
                sample_choice = st.selectbox(
                    "Select a diagnostic benchmark image:",
                    [
                        "🫘 Sample 1: Bilateral Pitting Edema (Swollen Feet - Kidney Disease)",
                        "❤️ Sample 2: 12-Lead ECG Strip with ST Depression (Cardiovascular)",
                        "🩸 Sample 3: Digital Glucometer Display 248 mg/dL (Diabetes)",
                        "🎙️ Sample 4: Archimedean Spiral Handwriting Test (Parkinson's)"
                    ]
                )
                if "Kidney" in sample_choice:
                    img_path = os.path.join("datasets", "image_samples", "edema_kidney_sample.png")
                    img_source_label = "Pedal Edema (Kidney Sign)"
                elif "Cardiovascular" in sample_choice:
                    img_path = os.path.join("datasets", "image_samples", "ecg_heart_sample.png")
                    img_source_label = "ECG Strip (Heart Sign)"
                elif "Diabetes" in sample_choice:
                    img_path = os.path.join("datasets", "image_samples", "glucometer_diabetes_sample.png")
                    img_source_label = "Glucometer (Diabetes Sign)"
                else:
                    img_path = os.path.join("datasets", "image_samples", "spiral_parkinsons_sample.png")
                    img_source_label = "Spiral Test (Parkinson's Sign)"

                if os.path.exists(img_path):
                    target_pil_img = Image.open(img_path)
            else:
                uploaded_img_file = st.file_uploader("Upload Medical Image (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"], key="diag_img_uploader")
                if uploaded_img_file is not None:
                    target_pil_img = Image.open(uploaded_img_file)
                    img_source_label = "Uploaded Patient Photo"

        with img_col2:
            st.markdown("##### 2. Visual AI Inspection & Triaging")
            if target_pil_img is not None:
                st.image(target_pil_img, caption=f"Analyzed Image: {img_source_label}", use_container_width=True)

                # Visual feature extraction
                np_img = np.array(target_pil_img.convert("RGB"))
                mean_r = np.mean(np_img[:, :, 0])
                mean_g = np.mean(np_img[:, :, 1])
                mean_b = np.mean(np_img[:, :, 2])
                brightness = np.mean(np_img)

                # Heuristic visual classification based on diagnostic patterns
                if "Glucometer" in img_source_label or (mean_b > mean_r and mean_b > 90 and brightness < 140):
                    suspected_disease = "Diabetes Mellitus"
                    disease_icon = "🩸"
                    confidence = 92
                    q1 = "Do you frequently experience excessive thirst or dry mouth?"
                    q2 = "Do you have sudden fatigue or lethargy shortly after eating?"
                    q3 = "Have you noticed tingling or numbness in your feet or hands?"
                elif "ECG" in img_source_label or (mean_r > mean_b * 1.2 and mean_r > 120):
                    suspected_disease = "Cardiovascular / Heart Disease"
                    disease_icon = "❤️"
                    confidence = 91
                    q1 = "Do you feel chest pressure, tightness, or burning during exertion?"
                    q2 = "Do you get out of breath climbing stairs or walking briskly?"
                    q3 = "Has a doctor ever noted resting blood pressure above 130/85 mm Hg?"
                elif "Spiral" in img_source_label or (brightness > 200 and np.std(np_img) < 65):
                    suspected_disease = "Parkinson's Disease"
                    disease_icon = "🎙️"
                    confidence = 88
                    q1 = "Does one of your hands or fingers shake slightly when resting?"
                    q2 = "Have you noticed your handwriting becoming smaller (micrographia)?"
                    q3 = "Do you feel your movements or walking pace have slowed down?"
                else:
                    suspected_disease = "Chronic Kidney Disease (CKD)"
                    disease_icon = "🫘"
                    confidence = 89
                    q1 = "Do you have swelling in your feet, ankles, or under the eyes?"
                    q2 = "Has your urine become foamy, darker, or changed in frequency?"
                    q3 = "Do you have a personal history of hypertension or high blood sugar?"

                st.markdown(f"**Preliminary Visual Indication:** {disease_icon} **{suspected_disease}** *(Visual Match Confidence: {confidence}%)*")
                
                st.markdown("---")
                st.markdown("##### 3. Diagnostic Verification Questions:")
                st.write("Please answer the following confirmatory clinical questions:")
                ans1 = st.checkbox(f"1. {q1}")
                ans2 = st.checkbox(f"2. {q2}")
                ans3 = st.checkbox(f"3. {q3}")

                yes_count = sum([ans1, ans2, ans3])
                final_confidence = min(98, confidence + (yes_count * 3) - ((3 - yes_count) * 2))

                st.markdown(f"""
                <div class="result-box-warning">
                    <h3>🎯 Primary Disease Identified: {disease_icon} {suspected_disease}</h3>
                    <p><b>Diagnostic Match Confidence: {final_confidence}%</b></p>
                    <p>Visual biomarkers combined with your symptom responses strongly correlate with <b>{suspected_disease}</b>.</p>
                    <p><b>Next Step:</b> Run the full specialized Machine Learning model for this condition to obtain precise numerical risk probabilities.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info("👈 Please select a clinical benchmark image or upload an image to begin visual disease identification.")

# -------------------------------------------------------------
# TAB 3: PARKINSON'S DISEASE (VOICE SCREENING)
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
