import streamlit as st
import numpy as np
import librosa
import joblib
import io
from audiorecorder import audiorecorder

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="🎙️ Voice Door Command Detector",
    page_icon="🎤",
    layout="centered"
)

# --- GLOBAL STYLE ---
st.markdown("""
<style>

/* FONT & BODY */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* CARD STYLE */
.result-card {
    margin-top: 25px;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    font-weight: 600;
    font-size: 1.4rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}

.result-open {
    background: linear-gradient(135deg, #d4edda, #b7dfc4);
    color: #0f5132;
}

.result-close {
    background: linear-gradient(135deg, #f8d7da, #f2b9bd);
    color: #842029;
}

.result-noise {
    background: linear-gradient(135deg, #ececec, #d9d9d9);
    color: #343a40;
}

/* HEADER TITLE */
.title {
    font-size: 2.3rem;
    font-weight: 700;
    text-align: center;
    padding-top: 10px;
}

/* DESCRIPTION TEXT */
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #5a5a5a;
    margin-bottom: 20px;
}

/* MIC BOX */
.mic-box {
    margin-top: 15px;
    background: #ffffff;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}

</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("speech_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        st.error("Model atau scaler tidak ditemukan. Jalankan training terlebih dahulu.")
        return None, None

model, scaler = load_models()
RMS_THRESHOLD = 0.005


# --- FEATURE EXTRACTION ---
def extract_features(y, sr=22050):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        rms_val = np.mean(librosa.feature.rms(y=y_trimmed))

        if rms_val < RMS_THRESHOLD:
            return 'below_threshold'

        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y_trimmed))
        features = np.hstack((mfccs_mean, zcr_mean, rms_val))
        return features
    except:
        return None


# --- UI ---
st.markdown('<div class="title">🎙️ Voice Door Command Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ucapkan perintah “Buka” atau “Tutup”, lalu berhenti untuk analisis.</div>', unsafe_allow_html=True)

class_labels = {0: "close", 1: "noise", 2: "open"}

if model is not None:

    st.markdown('<div class="mic-box">', unsafe_allow_html=True)
    audio = audiorecorder("🎤 Klik untuk Merekam", "⏳ Merekam... klik lagi untuk stop")
    st.markdown('</div>', unsafe_allow_html=True)

    status = st.empty()

    if len(audio) > 0:
        st.write("🔍 Sedang menganalisis audio…")

        audio_bytes = audio.export().read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

        features = extract_features(y, sr)

        if features is None or features == 'below_threshold':
            prediction_label = "noise"
        else:
            features_scaled = scaler.transform([features])
            prediction_idx = model.predict(features_scaled)[0]
            prediction_label = class_labels[prediction_idx]

        # TAMPILKAN HASIL DENGAN STYLE CARD LEBIH BAGUS
        if prediction_label == "open":
            status.markdown(
                f'<div class="result-card result-open">🔓 Opening the door now</div>',
                unsafe_allow_html=True
            )
        elif prediction_label == "close":
            status.markdown(
                f'<div class="result-card result-close">🔒 The door will be closed soon</div>',
                unsafe_allow_html=True
            )
        else:
            status.markdown(
                f'<div class="result-card result-noise">⚠️ Perintah tidak dikenali (Noise)</div>',
                unsafe_allow_html=True
            )

        st.audio(audio_bytes)

