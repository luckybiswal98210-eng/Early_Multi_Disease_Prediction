import numpy as np
import soundfile as sf
import os

def create_sample_audios():
    os.makedirs(os.path.join("datasets", "audio_samples"), exist_ok=True)
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Healthy sustained vowel AH (around 150 Hz base, formants at 750Hz and 1250Hz)
    f0 = 150.0
    sig_healthy = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 750 * t) + 0.2 * np.sin(2 * np.pi * 1250 * t)
    sig_healthy += 0.01 * np.random.normal(0, 1, len(t))
    sig_healthy = sig_healthy / np.max(np.abs(sig_healthy)) * 0.8
    h_path = os.path.join("datasets", "audio_samples", "sample_healthy_vowel.wav")
    sf.write(h_path, sig_healthy, sr)

    # 2. Parkinsonian dysarthric vowel AH (tremor modulation 5Hz + pitch jitter + noise)
    tremor_mod = 1.0 + 0.35 * np.sin(2 * np.pi * 5.0 * t)
    pitch_mod = f0 + 15.0 * np.sin(2 * np.pi * 4.5 * t)
    sig_pd = 0.4 * np.sin(2 * np.pi * pitch_mod * t) * tremor_mod + 0.25 * np.sin(2 * np.pi * 700 * t)
    sig_pd += 0.05 * np.random.normal(0, 1, len(t))
    sig_pd = sig_pd / np.max(np.abs(sig_pd)) * 0.8
    pd_path = os.path.join("datasets", "audio_samples", "sample_tremor_vowel.wav")
    sf.write(pd_path, sig_pd, sr)

    print(f"Generated demo audios:\n - {h_path}\n - {pd_path}")

if __name__ == "__main__":
    create_sample_audios()
