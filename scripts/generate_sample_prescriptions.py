import os
from PIL import Image, ImageDraw, ImageFont

def create_sample_prescriptions():
    output_dir = os.path.join("datasets", "image_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    prescriptions = [
        {
            "filename": "prescription_diabetes.png",
            "title": "METABOLIC & ENDOCRINE CARE CLINIC",
            "patient": "Patient: Rajesh Kumar | Age: 52 Yrs | Sex: Male | Date: 14-Oct-2026",
            "diagnosis": "Clinical Diagnosis: Type 2 Diabetes Mellitus with Metabolic Syndrome",
            "rx": [
                "1. Tab. METFORMIN HCl 500 mg — 1 Tab Twice Daily (After Meals)",
                "2. Tab. GLIMEPIRIDE 2 mg — 1 Tab Once Daily (Before Breakfast)",
                "3. Tab. ATORVASTATIN 10 mg — 1 Tab at Bedtime (HS)",
                "Advice: Fasting Blood Sugar (FBS) & HbA1c test after 4 weeks.",
                "Diet: Low glycemic index diet, strict sugar avoidance, 30 min daily walking."
            ],
            "doctor": "Dr. S. K. Mukherjee, MD (Endocrinology) | Reg No: MED-78421"
        },
        {
            "filename": "prescription_heart.png",
            "title": "APOLLO HEART & CARDIOVASCULAR CENTRE",
            "patient": "Patient: Sunita Sharma | Age: 58 Yrs | Sex: Female | Date: 12-Oct-2026",
            "diagnosis": "Clinical Diagnosis: Coronary Artery Disease (CAD), Grade-2 Hypertension",
            "rx": [
                "1. Tab. TELMISARTAN 40 mg + AMLODIPINE 5 mg — 1 Tab OD Morning",
                "2. Tab. ATORVASTATIN 20 mg — 1 Tab at Bedtime (HS)",
                "3. Tab. ASPIRIN (Ecosprin) 75 mg — 1 Tab Post Lunch",
                "4. Tab. METOPROLOL SUCCINATE 25 mg — 1 Tab OD",
                "Advice: 12-Lead ECG & 2D-Echocardiography. Monitor BP daily.",
                "Precautions: Low sodium diet (<2g/day), avoid strenuous exertion."
            ],
            "doctor": "Dr. Arun K. Verma, MD, DM (Cardiology) | Reg No: CAR-99120"
        },
        {
            "filename": "prescription_kidney.png",
            "title": "NATIONAL INSTITUTE OF NEPHROLOGY & RENAL CARE",
            "patient": "Patient: Ramesh Patel | Age: 61 Yrs | Sex: Male | Date: 16-Oct-2026",
            "diagnosis": "Clinical Diagnosis: Chronic Kidney Disease (CKD Stage 3b) with Fluid Retention",
            "rx": [
                "1. Tab. TORSEMIDE 10 mg — 1 Tab in Morning (Diuretic for Pedal Edema)",
                "2. Tab. SODIUM BICARBONATE 500 mg — 1 Tab Twice Daily",
                "3. Tab. CALCIUM ACETATE 667 mg — 1 Tab with meals (Phosphate binder)",
                "4. Tab. FEBUXOSTAT 40 mg — 1 Tab Once Daily",
                "Advice: Serum Creatinine, Blood Urea, Urine Routine & Albumin test.",
                "Diet: Renal diet (Low Potassium, Low Phosphorus, Strict Salt Restriction)."
            ],
            "doctor": "Dr. Meenakshi Sundaram, MD, DM (Nephrology) | Reg No: NEP-43180"
        },
        {
            "filename": "prescription_parkinsons.png",
            "title": "NEUROLOGICAL SCIENCES & MOVEMENT DISORDERS CLINIC",
            "patient": "Patient: Harish Chandra | Age: 66 Yrs | Sex: Male | Date: 18-Oct-2026",
            "diagnosis": "Clinical Diagnosis: Parkinson's Disease (Idiopathic Parkinsonism, Tremor Predominant)",
            "rx": [
                "1. Tab. SYNDOPA PLUS (Levodopa 100mg + Carbidopa 25mg) — 1/2 Tab TDS x 1 wk, then 1 Tab TDS",
                "2. Tab. PACITANE (Trihexyphenidyl) 2 mg — 1 Tab Twice Daily",
                "3. Tab. PRAMIPEXOLE 0.375 mg — 1 Tab at Bedtime",
                "Advice: Neuro-physiotherapy, speech clarity exercises, acoustic voice monitoring.",
                "Precautions: Take Levodopa 1 hour before or 2 hours after high-protein meals."
            ],
            "doctor": "Dr. Pradeep R. Menon, MD, DM (Neurology) | Reg No: NEU-61245"
        }
    ]
    
    for item in prescriptions:
        img = Image.new('RGB', (850, 520), color='#ffffff')
        draw = ImageDraw.Draw(img)
        
        # Border
        draw.rectangle([10, 10, 840, 510], outline='#0284c7', width=3)
        draw.rectangle([15, 15, 835, 505], outline='#e0f2fe', width=1)
        
        # Header banner
        draw.rectangle([18, 18, 832, 75], fill='#0369a1')
        draw.text((35, 32), "🏥 " + item["title"], fill="#ffffff")
        
        # Patient details bar
        draw.rectangle([18, 85, 832, 120], fill='#f0f9ff')
        draw.text((35, 95), item["patient"], fill="#0369a1")
        
        # Diagnosis
        draw.rectangle([18, 130, 832, 165], fill='#fef2f2')
        draw.text((35, 140), "📋 " + item["diagnosis"], fill="#b91c1c")
        
        # Rx Symbol
        draw.text((35, 175), "℞ PRESCRIPTION & MEDICATIONS:", fill="#0f172a")
        
        # Medications list
        y_pos = 205
        for line in item["rx"]:
            draw.text((50, y_pos), line, fill="#1e293b")
            y_pos += 30
            
        # Doctor signature line
        draw.line([50, 460, 800, 460], fill='#cbd5e1', width=1)
        draw.text((50, 470), "Attending Physician: " + item["doctor"], fill="#475569")
        
        save_path = os.path.join(output_dir, item["filename"])
        img.save(save_path)
        print("Saved prescription template:", save_path)

if __name__ == "__main__":
    create_sample_prescriptions()
