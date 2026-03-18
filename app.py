import streamlit as st
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Vocal Mood Intelligence", layout="wide")

# Professional CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    /* Metric Visibility Fix */
    [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #374151 !important; font-weight: 500 !important; }
    .stMetric { 
        background-color: #ffffff; padding: 15px; border-radius: 10px; 
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05); border: 1px solid #d1d5db;
    }
    .mood-box { 
        padding: 25px; border-radius: 12px; text-align: center; 
        font-size: 32px; font-weight: bold; color: white; margin-bottom: 20px; 
        text-transform: uppercase; letter-spacing: 2px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
st.sidebar.title("🎙️ System Control")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.success("Engine Mode: Simulation (Local)")

# --- 3. ANALYTICS FUNCTIONS ---

def extract_pro_features(audio_path):
    """ Extracts acoustic features for patent-grade reporting """
    y, sr = librosa.load(audio_path, duration=3)
    y_trimmed, _ = librosa.effects.trim(y)
    y_filt = librosa.effects.preemphasis(y_trimmed)
    
    mfcc = librosa.feature.mfcc(y=y_filt, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y_filt))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_filt))
    
    features = np.hstack([mfcc_mean, [rms, centroid]])
    cols = [f'mfcc_{i}' for i in range(40)] + ['rms_energy', 'spectral_centroid']
    return pd.DataFrame([features], columns=cols), y_filt, sr

def get_prediction(conn, features_df):
    """ Using your requested simulation logic """
    if not conn:
        # Fallback Mock Logic
        time.sleep(1.5)
        moods = ['Happy', 'Angry', 'Sad', 'Neutral', 'Fearful']
        return np.random.choice(moods), np.random.uniform(85, 99), "SIMULATED"
    
    # This block is kept for structure but won't be reached in Simulation mode
    return "Error", 0, "No Cloud Engine Linked"

# --- 4. MAIN UI ---
st.title("🎙️ AI Vocal Mood Intelligence System")
st.markdown("##### Real-Time Performance Dashboard")
st.divider()

col1, col2 = st.columns([1, 1.5], gap="large")

# Force connection to None for local simulation
sas_conn = None

with col1:
    st.info("Upload a .wav file for spectral analysis.")
    uploaded_file = st.file_uploader("Choose Audio File", type=["wav"])
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("🚀 EXECUTE AI ANALYSIS"):
            with st.status("Analyzing Signal...", expanded=False) as status:
                # 1. Feature Extraction
                df, y, sr = extract_pro_features(uploaded_file)
                
                # 2. Prediction (Fixed argument order)
                pred, conf, engine = get_prediction(sas_conn, df)
                
                st.session_state.analysis = {
                    "pred": pred, "conf": conf, "engine": engine, 
                    "y": y, "sr": sr, "df": df
                }
                status.update(label="Analysis Complete", state="complete")

with col2:
    if 'analysis' in st.session_state:
        res = st.session_state.analysis
        color_map = {
            'Angry': '#ff4b4b', 'Happy': '#28a745', 
            'Sad': '#6c757d', 'Neutral': '#17a2b8', 'Fearful': '#ffc107'
        }
        bg_color = color_map.get(res['pred'], '#333')
        
        # Mood Result Box
        st.markdown(f'<div class="mood-box" style="background-color: {bg_color};">{res["pred"]}</div>', unsafe_allow_html=True)
        
        # Visible Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Confidence Score", f"{res['conf']:.2f}%")
        with m2:
            st.metric("Inference Engine", res['engine'])
        
        # Visualizations
        tab1, tab2 = st.tabs(["🌈 Mel-Spectrogram", "📊 Feature Analysis"])
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=res['y'], sr=res['sr'])
            img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, format='%+2.0f dB')
            st.pyplot(fig)
        with tab2:
            st.write("**Top MFCC Coefficients (Acoustic Fingerprint):**")
            st.dataframe(res['df'].iloc[:, :12], use_container_width=True)
    else:
        st.write("Awaiting acoustic input... Upload a file to begin.")