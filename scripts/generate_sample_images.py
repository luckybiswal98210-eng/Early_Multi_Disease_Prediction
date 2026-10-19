import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images():
    output_dir = os.path.join("datasets", "image_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ECG Strip / Cardiac Waveform (Cardiovascular)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    fig.patch.set_facecolor('#fff1f2')
    ax.set_facecolor('#ffe4e6')
    ax.grid(True, color='#fca5a5', linestyle='-', linewidth=0.5)
    
    # Synthetic ECG signal with ST depression
    t = np.linspace(0, 4, 1000)
    ecg = np.zeros_like(t)
    for i in range(4):
        t_center = i + 0.5
        # P wave
        ecg += 0.15 * np.exp(-((t - (t_center - 0.2)) ** 2) / 0.002)
        # QRS complex
        ecg -= 0.15 * np.exp(-((t - (t_center - 0.05)) ** 2) / 0.0005)
        ecg += 1.0 * np.exp(-((t - t_center) ** 2) / 0.0005)
        ecg -= 0.25 * np.exp(-((t - (t_center + 0.05)) ** 2) / 0.0005)
        # ST depression & T wave
        ecg -= 0.18 * np.exp(-((t - (t_center + 0.15)) ** 2) / 0.005)
        ecg += 0.25 * np.exp(-((t - (t_center + 0.3)) ** 2) / 0.004)
        
    ax.plot(t, ecg, color='#be123c', linewidth=2)
    ax.set_title("Clinical ECG Trace - ST Segment Depression (Cardiac Ischemia)", fontsize=10, color='#9f1239', pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#fca5a5')
    plt.tight_layout()
    ecg_path = os.path.join(output_dir, "ecg_heart_sample.png")
    plt.savefig(ecg_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    # 2. Glucometer Digital Display (Diabetes)
    img_gl = Image.new('RGB', (600, 300), color='#0f172a')
    draw = ImageDraw.Draw(img_gl)
    # Bezel
    draw.rounded_rectangle([30, 20, 570, 280], radius=24, outline='#38bdf8', width=4, fill='#1e293b')
    # LCD Screen
    draw.rounded_rectangle([60, 50, 540, 220], radius=16, outline='#64748b', width=2, fill='#0284c7')
    # Display text
    draw.text((80, 70), "DIGITAL GLUCOSE MONITOR", fill="#e0f2fe")
    draw.text((120, 100), "248", fill="#ffffff")
    draw.text((380, 130), "mg / dL", fill="#bae6fd")
    draw.text((80, 185), "HIGH - POSTPRANDIAL HYPERGLYCEMIA", fill="#fef08a")
    gl_path = os.path.join(output_dir, "glucometer_diabetes_sample.png")
    img_gl.save(gl_path)
    
    # 3. Archimedean Spiral Handwriting Test (Parkinson's Tremor)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    theta = np.linspace(0, 8 * np.pi, 1000)
    r = 0.5 * theta
    # Add tremor noise
    r_tremor = r + 0.4 * np.sin(18 * theta) * np.random.uniform(0.5, 1.2, len(theta))
    x = r_tremor * np.cos(theta)
    y = r_tremor * np.sin(theta)
    ax.plot(x, y, color='#6366f1', linewidth=1.5, alpha=0.85)
    ax.set_title("Archimedean Spiral Test - Micrographia & Resting Tremor", fontsize=9, color='#312e81', pad=10)
    ax.axis('off')
    plt.tight_layout()
    spiral_path = os.path.join(output_dir, "spiral_parkinsons_sample.png")
    plt.savefig(spiral_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    # 4. Pedal Fluid Retention / Edema (Kidney Disease)
    img_ed = Image.new('RGB', (600, 300), color='#f1f5f9')
    draw_ed = ImageDraw.Draw(img_ed)
    draw_ed.rounded_rectangle([20, 20, 580, 280], radius=16, outline='#cbd5e1', width=3, fill='#ffffff')
    draw_ed.text((40, 40), "CLINICAL PHOTO: BILATERAL PITTING EDEMA", fill="#0f172a")
    draw_ed.text((40, 75), "Observation: Grade 3+ Pitting Ankle Swelling", fill="#b91c1c")
    draw_ed.text((40, 110), "Physiological Marker: Systemic Sodium/Fluid Retention", fill="#334155")
    draw_ed.text((40, 140), "Differential Diagnosis: Chronic Kidney Disease (Renal Failure)", fill="#0369a1")
    draw_ed.text((40, 180), "Associated Lab Biomarkers:\n- Serum Creatinine > 1.8 mg/dL\n- Urine Albumin +++\n- Estimated GFR < 45 mL/min", fill="#475569")
    ed_path = os.path.join(output_dir, "edema_kidney_sample.png")
    img_ed.save(ed_path)
    
    print("Generated 4 sample medical diagnostic images in datasets/image_samples/")

if __name__ == "__main__":
    generate_sample_images()
