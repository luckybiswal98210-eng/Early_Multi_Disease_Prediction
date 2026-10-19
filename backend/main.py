import os
import tempfile
import numpy as np
import joblib
import librosa
from gtts import gTTS
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import io
from PIL import Image
import sys

# Add parent directory to path to import clinical_vision if available
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import clinical_vision
except ImportError:
    clinical_vision = None

app = FastAPI(
    title="Intelligent Healthcare Multi-Disease Prediction API",
    description="Production Machine Learning Microservice for Cardiovascular, Parkinson's, Diabetes, and Chronic Kidney Disease screening.",
    version="2.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# LOAD TRAINED SCIKIT-LEARN MODELS & SCALERS
# -------------------------------------------------------------
MODELS_DIR = os.path.join(parent_dir, "models")

try:
    park_model = joblib.load(os.path.join(MODELS_DIR, "audio_parkinsons_model.pkl"))
    park_scaler = joblib.load(os.path.join(MODELS_DIR, "audio_scaler.pkl"))

    heart_model = joblib.load(os.path.join(MODELS_DIR, "heart_model.pkl"))
    heart_scaler = joblib.load(os.path.join(MODELS_DIR, "heart_scaler.pkl"))

    diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))

    kidney_model = joblib.load(os.path.join(MODELS_DIR, "kidney_model.pkl"))
    kidney_scaler = joblib.load(os.path.join(MODELS_DIR, "kidney_scaler.pkl"))
    models_ready = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_ready = False

# -------------------------------------------------------------
# MULTI-LANGUAGE TRANSLATION DICTIONARY
# -------------------------------------------------------------
PARKINSONS_TRANSLATIONS = {
    "en": {
        "healthy": "Voice acoustic analysis indicates healthy vocal stability with {prob} percent probability. Vocal harmonics are within normal limits. No significant signs of Parkinson's detected.",
        "risk": "Voice acoustic analysis detected elevated probability of Parkinsonian dysarthria at {prob} percent. Clinical consultation with a specialist neurologist is strongly recommended."
    },
    "hi": {
        "healthy": "आवाज ध्वनिक विश्लेषण से पता चलता है कि स्वर स्थिरता सामान्य है और पार्किंसन की संभावना केवल {prob} प्रतिशत है। कोई गंभीर लक्षण नहीं मिले।",
        "risk": "आवाज विश्लेषण में {prob} प्रतिशत पार्किंसन डिस्आर्थ्रिया का उच्च जोखिम पाया गया है। कृपया जल्द से जल्द न्यूरोलॉजिस्ट डॉक्टर से परामर्श करें।"
    },
    "bn": {
        "healthy": "ভয়েস বিশ্লেষণ অনুসারে স্বর স্থিতিশীলতা স্বাভাবিক এবং পার্কিনসনের সম্ভাবনা মাত্র {prob} শতাংশ। কোনো গুরুতর লক্ষণ পাওয়া যায়নি।",
        "risk": "ভয়েস বিশ্লেষণে {prob} শতাংশ পার্কিনসন্স ঝুঁকির লক্ষণ ধরা পড়েছে। একজন বিশেষজ্ঞ নিউরোলজিস্টের সাথে অবিলম্বে পরামর্শ করার সুপারিশ করা হচ্ছে।"
    },
    "te": {
        "healthy": "వాయిస్ శబ్ద విశ్లేషణ ప్రకారం స్వర స్థిరత్వం సాధారణంగా ఉంది, మరియు పార్కిన్సన్స్ సంభావ్యత కేవలం {prob} శాతం మాత్రమే. ఎటువంటి ప్రమాదకర లక్షణాలు లేవు.",
        "risk": "వాయిస్ విశ్లేషణలో పార్కిన్సన్స్ డిసార్థ్రియా వచ్చే ప్రమాదం {prob} శాతం ఎక్కువగా గుర్తించబడింది. న్యూరాలజిస్ట్ వైద్యుడిని సంప్రదించాలని సిఫార్సు చేయబడింది."
    },
    "ta": {
        "healthy": "குரல் ஒலி பகுப்பாய்வு சாதாரண குரல் நிலைத்தன்மையைக் காட்டுகிறது. பார்கின்சன் நோய் வாய்ப்பு {prob} சதவீதம் மட்டுமே. தீவிர அறிகுறிகள் எதுவும் இல்லை.",
        "risk": "குரல் பகுப்பாய்வில் பார்கின்சன் நோய் ஆபத்து {prob} சதவீதமாக உயர்ந்துள்ளது. நரம்பியல் மருத்துவரை உடனடியாக அணுகுமாறு பரிந்துரைக்கப்படுகிறது."
    },
    "kn": {
        "healthy": "ಧ್ವನಿ ವಿಶ್ಲೇಷಣೆಯು ಸಾಮಾನ್ಯ ಧ್ವನಿ ಸ್ಥಿರತೆಯನ್ನು ತೋರಿಸುತ್ತದೆ, ಪಾರ್ಕಿನ್ಸನ್ ಸಾಧ್ಯತೆ ಕೇವಲ {prob} ಪ್ರತಿಶತ. ಯಾವುದೇ ಗಂಭೀರ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
        "risk": "ಧ್ವನಿ ವಿಶ್ಲೇಷಣೆಯಲ್ಲಿ ಪಾರ್ಕಿನ್ಸನ್ ಅಪಾಯದ ಸಾಧ್ಯತೆ {prob} ಪ್ರತಿಶತ ಹೆಚ್ಚಾಗಿರುವುದು ಕಂಡುಬಂದಿದೆ. ನರಶಾಸ್ತ್ರಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಲು ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ."
    },
    "ml": {
        "healthy": "ശബ്ദ വിശകലനം സാധാരണ ശബ്ദ സ്ഥിരത കാണിക്കുന്നു. പാർക്കിൻസൺസ് സാധ്യത {prob} ശതമാനം മാത്രമാണ്. ഗുരുതരമായ ലക്ഷണങ്ങളൊന്നും കണ്ടെത്തിയില്ല.",
        "risk": "ശബ്ദ വിശകലനത്തിൽ പാർക്കിൻസൺസ് രോഗ സാധ്യത {prob} ശതമാനമായി ഉയർന്നതായി കണ്ടെത്തി. ഒരു ന്യൂറോളജിസ്റ്റിനെ ഉടൻ സമീപിക്കാൻ ശുപാർശ ചെയ്യുന്നു."
    },
    "gu": {
        "healthy": "અવાજ વિશ્લેષણ સામાન્ય સ્વર સ્થિરતા દર્શાવે છે અને પાર્કિન્સનની સંભાવના માત્ર {prob} ટકા છે. કોઈ ગંભીર ચિહ્નો મળ્યા નથી.",
        "risk": "અવાજ વિશ્લેષણમાં {prob} ટકા પાર્કિન્સનનું ઊંચું જોખમ જણાયું છે. તાત્કાલિક ન્યુરોલોજીસ્ટ ડોક્ટરની સલાહ લેવાની ભલામણ કરવામાં આવે છે."
    },
    "mr": {
        "healthy": "आवाज ध्वनी विश्लेषणावरून स्वर स्थिरता सामान्य असल्याचे दिसून येते आणि पार्किन्सन्सची शक्यता केवळ {prob} टक्के आहे. कोणतीही गंभीर लक्षणे आढळली नाहीत.",
        "risk": "आवाज विश्लेषणात {prob} टक्के पार्किन्सन्सचा धोका जास्त असल्याचे आढळले आहे. न्यूरोलॉजिस्ट तज्ज्ञांचा सल्ला घेण्याची जोरदार शिफारस केली जाते."
    },
    "pa": {
        "healthy": "ਆਵਾਜ਼ ਵਿਸ਼ਲੇਸ਼ਣ ਤੋਂ ਪਤਾ ਲੱਗਦਾ ਹੈ ਕਿ ਆਵਾਜ਼ ਦੀ ਸਥਿਰਤਾ ਆਮ ਹੈ ਅਤੇ ਪਾਰਕਿੰਸਨ ਦੀ ਸੰਭਾਵਨਾ ਸਿਰਫ਼ {prob} ਫ਼ੀਸਦੀ ਹੈ। ਕੋਈ ਗੰਭੀਰ ਲੱਛਣ ਨਹੀਂ ਮਿਲੇ।",
        "risk": "ਆਵਾਜ਼ ਵਿਸ਼ਲੇਸ਼ਣ ਵਿੱਚ {prob} ਫ਼ੀਸਦੀ ਪਾਰਕਿੰਸਨ ਦਾ ਉੱਚ ਖ਼ਤਰਾ ਪਾਇਆ ਗਿਆ ਹੈ। ਨਿਊਰੋਲੋਜਿਸਟ ਡਾਕਟਰ ਦੀ ਸਲਾਹ ਲੈਣ ਦੀ ਸਿਫ਼ਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ।"
    },
    "es": {
        "healthy": "El análisis acústico de la voz indica estabilidad vocal saludable con una probabilidad de Parkinson de solo {prob} por ciento. No se detectan signos significativos.",
        "risk": "El análisis de voz detectó una probabilidad elevada de disartria parkinsoniana del {prob} por ciento. Se recomienda encarecidamente consultar a un neurólogo."
    },
    "fr": {
        "healthy": "L'analyse acoustique vocale indique une stabilité vocale saine avec une probabilité de Parkinson de seulement {prob} pour cent. Aucun signe significatif n'est détecté.",
        "risk": "L'analyse vocale a détecté une probabilité élevée de dysarthrie parkinsonienne de {prob} pour cent. Une consultation auprès d'un neurologue est fortement recommandée."
    },
    "de": {
        "healthy": "Die akustische Stimmanalyse zeigt eine gesunde Stimmstabilität mit einer Parkinson-Wahrscheinlichkeit von nur {prob} Prozent. Keine signifikanten Anzeichen festgestellt.",
        "risk": "Die Stimmanalyse hat eine erhöhte Wahrscheinlichkeit für Parkinson-Dysarthrie von {prob} Prozent festgestellt. Eine neurologische Fachberatung wird dringend empfohlen."
    }
}

# -------------------------------------------------------------
# PYDANTIC REQUEST MODELS
# -------------------------------------------------------------
class HeartRequest(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class DiabetesRequest(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    dpf: float
    age: float

class KidneyRequest(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    bgr: float
    bu: float
    sc: float
    hemo: float
    htn: int
    dm: int

class PrescriptionTextRequest(BaseModel):
    text: str

class TtsRequest(BaseModel):
    text: str
    lang: str = "en"

# -------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "service": "Intelligent Healthcare Multi-Disease Prediction API",
        "status": "online",
        "models_ready": models_ready,
        "endpoints": [
            "/api/predict/parkinsons",
            "/api/predict/heart",
            "/api/predict/diabetes",
            "/api/predict/kidney",
            "/api/vision/prescription",
            "/api/tts"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": models_ready}

# 1. PARKINSON'S VOICE ACOUSTIC ANALYSIS
@app.post("/api/predict/parkinsons")
async def predict_parkinsons(
    file: UploadFile = File(...),
    lang: str = Form("en"),
    symptoms_count: int = Form(0)
):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract 20 MFCCs, Spectral Centroid, ZCR, RMS
        y, sr = librosa.load(tmp_path, sr=22050)
        os.remove(tmp_path)

        if y.size == 0 or np.sqrt(np.mean(y**2)) < 0.005:
            return {
                "error": "Audio file appears to be silent. Please provide an audible sustained vowel 'AH' recording."
            }

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0].mean()
        rms = librosa.feature.rms(y=y)[0].mean()
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        feats = np.hstack([mfcc, zcr, rms, spec_centroid])

        scaled_feats = park_scaler.transform([feats])
        prob = float(park_model.predict_proba(scaled_feats)[0, 1])
        prob_pct = round(prob * 100, 1)

        is_detected = prob_pct >= 80.0
        status = "Parkinson's Detected" if is_detected else "Healthy Voice"

        # Generate native translated text
        lang_dict = PARKINSONS_TRANSLATIONS.get(lang, PARKINSONS_TRANSLATIONS["en"])
        template = lang_dict["risk"] if is_detected else lang_dict["healthy"]
        spoken_text = template.format(prob=int(prob_pct))

        # Generate base64 MP3 TTS audio
        audio_base64 = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                tts = gTTS(text=spoken_text, lang=lang, slow=False)
                tts.save(audio_tmp.name)
                audio_tmp.seek(0)
                audio_bytes = audio_tmp.read()
            os.remove(audio_tmp.name)
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            print(f"TTS generation error: {e}")

        return {
            "disease": "Parkinson's Disease",
            "probability": prob_pct,
            "status": status,
            "is_detected": is_detected,
            "symptoms_count": symptoms_count,
            "spoken_text": spoken_text,
            "lang": lang,
            "audio_base64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. CARDIOVASCULAR PREDICTION
@app.post("/api/predict/heart")
def predict_heart(req: HeartRequest):
    try:
        data = np.array([[
            req.age, req.sex, req.cp, req.trestbps, req.chol, req.fbs,
            req.restecg, req.thalach, req.exang, req.oldpeak, req.slope, req.ca, req.thal
        ]])
        scaled = heart_scaler.transform(data)
        prob = float(heart_model.predict_proba(scaled)[0, 1])
        prob_pct = round(prob * 100, 1)
        is_detected = prob >= 0.5

        return {
            "disease": "Cardiovascular Disease",
            "probability": prob_pct,
            "status": "Elevated Cardiovascular Risk" if is_detected else "Low Cardiovascular Risk",
            "is_detected": is_detected,
            "vitals": f"BP: {req.trestbps} mm Hg, Chol: {req.chol} mg/dL, HR: {req.thalach} bpm"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. DIABETES MELLITUS PREDICTION
@app.post("/api/predict/diabetes")
def predict_diabetes(req: DiabetesRequest):
    try:
        data = np.array([[
            req.pregnancies, req.glucose, req.blood_pressure, req.skin_thickness,
            req.insulin, req.bmi, req.dpf, req.age
        ]])
        scaled = diabetes_scaler.transform(data)
        prob = float(diabetes_model.predict_proba(scaled)[0, 1])
        prob_pct = round(prob * 100, 1)
        is_detected = prob >= 0.5

        return {
            "disease": "Diabetes Mellitus",
            "probability": prob_pct,
            "status": "Diabetic Risk Detected" if is_detected else "Non-Diabetic / Normal",
            "is_detected": is_detected,
            "vitals": f"Glucose: {req.glucose} mg/dL, BMI: {req.bmi}, Age: {req.age}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. CHRONIC KIDNEY DISEASE PREDICTION
@app.post("/api/predict/kidney")
def predict_kidney(req: KidneyRequest):
    try:
        data = np.array([[
            req.age, req.bp, req.sg, req.al, req.su, req.bgr, req.bu, req.sc,
            req.hemo, req.htn, req.dm
        ]])
        scaled = kidney_scaler.transform(data)
        prob = float(kidney_model.predict_proba(scaled)[0, 1])
        prob_pct = round(prob * 100, 1)
        is_detected = prob >= 0.5

        return {
            "disease": "Chronic Kidney Disease",
            "probability": prob_pct,
            "status": "CKD Detected" if is_detected else "Healthy Kidneys",
            "is_detected": is_detected,
            "vitals": f"Creatinine: {req.sc} mg/dL, Urea: {req.bu} mg/dL, Albumin: {req.al}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. CLINICAL PRESCRIPTION PARSER (NLP)
@app.post("/api/vision/prescription")
async def parse_prescription(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None)
):
    try:
        text = raw_text or ""
        if file is not None:
            content = await file.read()
            pil_img = Image.open(io.BytesIO(content))
            if clinical_vision:
                extracted, _ = clinical_vision.extract_text_from_image(pil_img, file.filename)
                text = (text + " " + extracted).strip()

        if clinical_vision:
            res = clinical_vision.parse_and_understand_prescription(text)
            return res
        return {"status": "error", "message": "Clinical vision module not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. TEXT-TO-SPEECH (gTTS)
@app.post("/api/tts")
def text_to_speech(req: TtsRequest):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
            tts = gTTS(text=req.text, lang=req.lang, slow=False)
            tts.save(audio_tmp.name)
            audio_tmp.seek(0)
            audio_bytes = audio_tmp.read()
        os.remove(audio_tmp.name)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_base64": audio_b64, "lang": req.lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
