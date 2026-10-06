"""
🔬 Skin Lesion Analyzer — Skin Cancer Detection Web App
=======================================================
A modern, fully-responsive, glassmorphism-styled Streamlit application
that classifies dermoscopic lesion images as benign or malignant using
a deep learning model (EfficientNet-style Keras model, 224x224 input).

Model weights are auto-downloaded from Google Drive via gdown.
"""

import os
import time
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ----------------------------------------------------------------------------
# 1. CORE CONFIGURATION & BACKEND
# ----------------------------------------------------------------------------

APP_TITLE = "🔬 Skin Lesion Analyzer"
CLASS_LABELS = ["benign", "malignant"]
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.keras"
GDRIVE_FILE_ID = "16tlVBG4-pbqmnWCUEaHCxtNCmQ_Y5lnS"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

st.set_page_config(
    page_title="Skin Lesion Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Download model weights from Google Drive (cached) and load the model."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model weights — please wait..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 and normalize to [0, 1] (standard TF pipeline)."""
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)  # add batch dimension


def predict(model, img_batch) -> tuple[str, float, np.ndarray]:
    """Return (label, confidence, full probability vector)."""
    probs = model.predict(img_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs


# ----------------------------------------------------------------------------
# 2. GLASSMORPHISM CSS — frosted glass, blur, animations, responsive layout
# ----------------------------------------------------------------------------

GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

/* ---------- Global ---------- */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(ellipse at 20% 20%, #1e3a5f 0%, #0b1020 60%, #060913 100%);
    color: #e8edf5;
}
[data-testid="stAppViewContainer"] {
    background-attachment: fixed;
}
.block-container { max-width: 1200px; padding-top: 2rem; }

/* ---------- Frosted glass base ---------- */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    padding: 1.5rem;
    transition: all 0.3s ease;
}
.glass:hover { transform: scale(1.02); }

/* ---------- Keyframes ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.7s ease both; }
.fade-in-slow { animation: fadeIn 1.1s ease both; }

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(96, 165, 250, 0.35); }
    50%      { box-shadow: 0 0 26px rgba(96, 165, 250, 0.65); }
}

/* ---------- Header banner ---------- */
.glass-header {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 24px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    padding: 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.8s ease both, pulseGlow 4s ease-in-out infinite;
}
.glass-header h1 { font-weight: 800; letter-spacing: -0.5px; margin: 0; font-size: clamp(1.6rem, 4vw, 2.6rem); }
.glass-header p { opacity: 0.85; margin: 0.4rem 0 0 0; font-weight: 300; }

.badge-disclaimer {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: rgba(251, 191, 36, 0.18);
    border: 1px solid rgba(251, 191, 36, 0.45);
    color: #fcd34d;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ---------- Result badges ---------- */
.badge {
    display: inline-block;
    padding: 0.7rem 2rem;
    border-radius: 999px;
    font-size: clamp(1.1rem, 3vw, 1.6rem);
    font-weight: 800;
    letter-spacing: 0.5px;
    animation: fadeIn 0.6s ease both;
}
.badge-benign {
    background: rgba(34, 197, 94, 0.22);
    border: 1px solid rgba(34, 197, 94, 0.6);
    color: #4ade80;
    box-shadow: 0 0 22px rgba(34, 197, 94, 0.35);
}
.badge-malignant {
    background: rgba(239, 68, 68, 0.22);
    border: 1px solid rgba(239, 68, 68, 0.6);
    color: #f87171;
    box-shadow: 0 0 22px rgba(239, 68, 68, 0.35);
}
.badge-uncertain {
    background: rgba(245, 158, 11, 0.22);
    border: 1px solid rgba(245, 158, 11, 0.6);
    color: #fbbf24;
    box-shadow: 0 0 22px rgba(245, 158, 11, 0.35);
}

/* ---------- Probability bar ---------- */
.prob-track {
    width: 100%;
    height: 16px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    overflow: hidden;
    margin: 0.4rem 0 1rem 0;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}

/* ---------- Sidebar glass ---------- */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}
[data-testid="stSidebar"] .glass { background: rgba(255, 255, 255, 0.06); }

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border: 1.5px dashed rgba(255, 255, 255, 0.35);
    border-radius: 18px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    transform: scale(1.02);
    border-color: rgba(96, 165, 250, 0.8);
    box-shadow: 0 0 24px rgba(96, 165, 250, 0.25);
}

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    background: rgba(96, 165, 250, 0.25);
    color: #e8edf5;
    font-weight: 700;
    padding: 0.6rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.02);
    background: rgba(96, 165, 250, 0.45);
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.4);
}

/* ---------- Spinner ---------- */
.stSpinner > div { border-top-color: #60a5fa !important; }

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover { transform: scale(1.02); }

/* ---------- Responsive helpers ---------- */
.responsive-img img {
    width: 100%;
    height: auto;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

/* checklist */
.checklist li { margin: 0.35rem 0; line-height: 1.5; }
.checklist { padding-left: 1.1rem; }

hr.soft {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    margin: 1.2rem 0;
}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. RESPONSIVE SIDEBAR — threshold slider + image guidelines checklist
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        help="Malignant probability above this threshold → classified as Malignant.",
    )
    st.caption(f"Current threshold: **{threshold:.2f}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass fade-in-slow" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown("### 📋 Image Guidelines")
    st.markdown(
        """
        <ul class="checklist">
            <li>✅ Use a <b>clear, in-focus</b> close-up of the lesion</li>
            <li>✅ Even, bright lighting — avoid harsh shadows</li>
            <li>✅ Fill most of the frame with the lesion</li>
            <li>✅ Formats: <b>.jpg</b>, <b>.jpeg</b>, or <b>.png</b></li>
            <li>⚠️ No blurry, filtered, or heavily compressed images</li>
            <li>⚠️ Hair, rulers, or ink marks may reduce accuracy</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="glass fade-in-slow" style="margin-top:1rem;">'
        "<small>🔒 Images are processed in-memory only and never stored.</small>"
        "</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# 4. GLASS HEADER & BANNER
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="glass-header">
        <h1>🔬 Skin Lesion Analyzer</h1>
        <p>AI-assisted dermoscopic image classification — Benign vs. Malignant</p>
        <span class="badge-disclaimer">⚕️ For research &amp; educational use only — not a medical diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Lazy-load model once per session (cached resource)
model = load_model()

# ----------------------------------------------------------------------------
# 5. INTERACTIVE UPLOAD AREA + IMAGE PREVIEW (responsive two-column layout)
# ----------------------------------------------------------------------------

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Lesion Image")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse — .jpg / .jpeg / .png",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="glass fade-in-slow" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown("#### 🩺 Ready to analyze")
        analyze_clicked = st.button("🔍 Analyze Lesion", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        analyze_clicked = False

with col_preview:
    st.markdown('<div class="glass fade-in responsive-img">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded lesion image", use_container_width=True)
        st.caption(f"Original size: {image.size[0]} × {image.size[1]} px → resized to 224 × 224")
    else:
        st.info("Upload an image to see a live preview here.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. RESULTS OUTPUT — badge, metrics, probability breakdown
# ----------------------------------------------------------------------------

if uploaded_file is not None and analyze_clicked:
    with st.spinner("⚙️ Running deep-learning inference..."):
        time.sleep(0.4)  # brief pause so the spinner is perceptible
        img_batch = preprocess_image(image)
        label, confidence, probs = predict(model, img_batch)
        benign_p, malignant_p = float(probs[0]), float(probs[1])

    # --- Decision logic with user-configurable threshold ---
    if malignant_p >= threshold:
        final_label, badge_class = "Malignant", "badge-malignant"
    elif benign_p >= threshold:
        final_label, badge_class = "Benign", "badge-benign"
    else:
        final_label, badge_class = "Uncertain — consult a dermatologist", "badge-uncertain"

    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.markdown("## 🧾 Analysis Results")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            f'<span class="badge {badge_class}">{"✅" if final_label=="Benign" else "🚨" if final_label.startswith("Malignant") else "⚠️"} {final_label.upper()}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with c3:
        st.metric("Threshold", f"{threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass fade-in-slow" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown("### 📊 Probability Breakdown")

    # Benign bar
    b_color = "#22c55e" if benign_p >= malignant_p else "#64748b"
    st.markdown(f"**🟢 Benign — {benign_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{benign_p * 100:.1f}%; background:{b_color};"></div></div>',
        unsafe_allow_html=True,
    )

    # Malignant bar
    m_color = "#ef4444" if malignant_p > benign_p else "#64748b"
    st.markdown(f"**🔴 Malignant — {malignant_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{malignant_p * 100:.1f}%; background:{m_color};"></div></div>',
        unsafe_allow_html=True,
    )

    if final_label.startswith("Malignant"):
        st.warning("🚨 This lesion shows features consistent with **malignancy**. Please consult a dermatologist promptly.")
    elif final_label.startswith("Uncertain"):
        st.info("⚠️ The model is uncertain about this lesion. Consider professional evaluation.")
    else:
        st.success("✅ This lesion appears **benign**. Continue routine skin monitoring.")

    st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is not None:
    st.markdown(
        '<div class="glass fade-in"><p style="text-align:center; opacity:0.8;">'
        "👆 Press <b>Analyze Lesion</b> to run the classifier.</p></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="glass fade-in"><p style="text-align:center; opacity:0.8;">'
        "📁 Upload a dermoscopic image to begin analysis.</p></div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="glass fade-in-slow" style="margin-top:2rem; text-align:center;">
        <small>
        ⚕️ <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
        It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for any skin concerns.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
