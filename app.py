"""
🔬 Skin Lesion Analyzer — Skin Cancer Detection Web App  (v2.0)
===============================================================
Redesigned UI:
• Fresh teal/cyan medical color scheme on deep navy
• Fixed glass top-navbar with smooth-scroll section links (page is never hidden)
• Extra animations: animated gradient title, floating aurora orbs, staggered
  card reveals, shimmer buttons, animated probability bars
• Fully responsive — mobile hamburger navbar, stacked columns, touch targets
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
    initial_sidebar_state="collapsed",  # navbar replaces sidebar on all devices
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
    return np.expand_dims(img_array, axis=0)


def predict(model, img_batch):
    probs = model.predict(img_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs


# ----------------------------------------------------------------------------
# 2. THEME CSS — teal/cyan medical palette, glass, animations, responsive
# ----------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');

/* ============ DESIGN TOKENS ============ */
/*  bg deep navy      : #071018 / #0a1622
    primary teal      : #14b8a6
    accent cyan       : #22d3ee
    soft mint         : #99f6e4
    glass surfaces    : rgba(20, 184, 166, 0.06–0.12)
    borders           : rgba(45, 212, 191, 0.25)
    text              : #d7f5f0                                    */

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(20,184,166,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 10%, rgba(34,211,238,0.10) 0%, transparent 55%),
        linear-gradient(160deg, #071018 0%, #0a1622 55%, #060d14 100%);
    color: #d7f5f0;
    scroll-behavior: smooth;
}
[data-testid="stAppViewContainer"] { background-attachment: fixed; }
.block-container { max-width: 1200px; padding-top: 6.5rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif; }

/* ============ FLOATING AURORA ORBS (ambient animation) ============ */
.orb {
    position: fixed; border-radius: 50%; filter: blur(90px);
    opacity: 0.35; pointer-events: none; z-index: 0;
    animation: drift 22s ease-in-out infinite alternate;
}
.orb-1 { width: 420px; height: 420px; background: #14b8a6; top: -120px; left: -120px; }
.orb-2 { width: 360px; height: 360px; background: #0e7490; bottom: -100px; right: -80px; animation-delay: -8s; }
.orb-3 { width: 260px; height: 260px; background: #22d3ee; top: 40%; left: 55%; opacity: 0.18; animation-delay: -14s; }
@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    50%  { transform: translate(60px, 40px) scale(1.12); }
    100% { transform: translate(-40px, 70px) scale(0.94); }
}

/* ============ FIXED TOP NAVBAR (replaces hidden-sidebar problem) ============ */
.topnav {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 2rem;
    background: rgba(7, 16, 24, 0.72);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(45, 212, 191, 0.22);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    animation: slideDown 0.6s ease both;
}
@keyframes slideDown { from { transform: translateY(-100%); } to { transform: translateY(0); } }
.topnav .brand {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem;
    color: #99f6e4; text-decoration: none; letter-spacing: 0.3px;
    display: flex; align-items: center; gap: 0.5rem; white-space: nowrap;
}
.topnav .brand .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #14b8a6; box-shadow: 0 0 12px #14b8a6;
    animation: blink 2.4s ease-in-out infinite;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
.topnav .links { display: flex; gap: 1.4rem; }
.topnav .links a {
    color: #b8e6de; text-decoration: none; font-weight: 600; font-size: 0.88rem;
    padding: 0.35rem 0.9rem; border-radius: 999px;
    border: 1px solid transparent; transition: all 0.3s ease; white-space: nowrap;
}
.topnav .links a:hover {
    color: #071018; background: linear-gradient(120deg, #14b8a6, #22d3ee);
    box-shadow: 0 0 18px rgba(34, 211, 238, 0.45);
}
.topnav .menu-btn {
    display: none; background: none; border: 1px solid rgba(45,212,191,0.4);
    color: #99f6e4; border-radius: 10px; font-size: 1.2rem;
    padding: 0.25rem 0.7rem; cursor: pointer; transition: all 0.3s ease;
}
/* hide Streamlit's default header & sidebar toggle — our navbar owns navigation */
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="collapsedControl"], [data-testid="stSidebar"] { display: none !important; }

/* ============ GLASS SURFACES ============ */
.glass {
    position: relative; z-index: 1;
    background: rgba(20, 184, 166, 0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(45, 212, 191, 0.22);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(153, 246, 228, 0.12);
    padding: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.glass:hover {
    transform: translateY(-4px);
    border-color: rgba(45, 212, 191, 0.45);
    box-shadow: 0 14px 40px rgba(20, 184, 166, 0.18), inset 0 1px 0 rgba(153, 246, 228, 0.18);
}

/* ============ ANIMATIONS ============ */
@keyframes fadeUp { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn   { 0% { opacity: 0; transform: scale(0.85); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes floatY  { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes gradientMove { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

.fade-up      { animation: fadeUp 0.7s ease both; }
.fade-up-1    { animation: fadeUp 0.7s ease 0.12s both; }
.fade-up-2    { animation: fadeUp 0.7s ease 0.24s both; }
.fade-up-3    { animation: fadeUp 0.7s ease 0.36s both; }
.pop-in       { animation: popIn 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
.float-anim   { animation: floatY 4s ease-in-out infinite; }

/* animated gradient headline */
.grad-text {
    background: linear-gradient(90deg, #99f6e4, #22d3ee, #14b8a6, #67e8f9, #99f6e4);
    background-size: 300% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 6s linear infinite;
}

/* ============ HERO / HEADER ============ */
.hero {
    text-align: center; padding: 2.2rem 1.5rem 1.8rem 1.5rem; margin-bottom: 1.6rem;
    overflow: hidden;
}
.hero h1 {
    font-size: clamp(1.8rem, 5vw, 3rem); font-weight: 800; letter-spacing: -1px; margin: 0;
}
.hero p.sub { opacity: 0.85; font-weight: 300; margin: 0.5rem 0 0 0; font-size: clamp(0.9rem, 2.5vw, 1.1rem); }
.hero .badge-disclaimer {
    display: inline-block; margin-top: 1rem; padding: 0.4rem 1.1rem; border-radius: 999px;
    background: rgba(251, 191, 36, 0.14); border: 1px solid rgba(251, 191, 36, 0.5);
    color: #fcd34d; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.4px;
    animation: popIn 0.7s ease 0.4s both;
}

/* ============ RESULT BADGES ============ */
.badge {
    display: inline-block; padding: 0.65rem 1.8rem; border-radius: 999px;
    font-size: clamp(1rem, 3vw, 1.5rem); font-weight: 800; letter-spacing: 0.5px;
    animation: popIn 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.badge-benign    { background: rgba(34, 197, 94, 0.18); border: 1px solid rgba(34,197,94,0.65);  color: #4ade80; box-shadow: 0 0 22px rgba(34,197,94,0.35); }
.badge-malignant { background: rgba(239, 68, 68, 0.18); border: 1px solid rgba(239,68,68,0.65);  color: #f87171; box-shadow: 0 0 22px rgba(239,68,68,0.35); }
.badge-uncertain { background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(245,158,11,0.65); color: #fbbf24; box-shadow: 0 0 22px rgba(245,158,11,0.35); }

/* ============ PROBABILITY BARS ============ */
.prob-track {
    width: 100%; height: 16px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.08); overflow: hidden; margin: 0.4rem 0 1.1rem 0;
    border: 1px solid rgba(255,255,255,0.08);
}
.prob-fill {
    height: 100%; border-radius: 999px; width: 0;
    transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
    animation: fadeUp 0.5s ease both;
}

/* ============ UPLOADER ============ */
[data-testid="stFileUploader"] section {
    background: rgba(20, 184, 166, 0.06);
    backdrop-filter: blur(12px);
    border: 1.5px dashed rgba(45, 212, 191, 0.4);
    border-radius: 18px; transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    transform: scale(1.015); border-color: #22d3ee;
    box-shadow: 0 0 26px rgba(34, 211, 238, 0.3);
}

/* ============ BUTTONS (shimmer) ============ */
.stButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(45, 212, 191, 0.4);
    background: linear-gradient(120deg, rgba(20,184,166,0.9), rgba(8,145,178,0.9), rgba(20,184,166,0.9));
    background-size: 200% auto;
    color: #04141a; font-weight: 800; padding: 0.7rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 24px rgba(20, 184, 166, 0.45);
    animation: shimmer 2s linear infinite;
}

/* ============ METRICS / SLIDERS ============ */
[data-testid="stMetric"] {
    background: rgba(20, 184, 166, 0.07); backdrop-filter: blur(10px);
    border: 1px solid rgba(45, 212, 191, 0.22); border-radius: 16px;
    padding: 0.9rem 1.1rem; transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(45,212,191,0.5); }
[data-testid="stMetricValue"] { color: #5eead4 !important; }
.stSlider label, .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #b8e6de !important; }

/* ============ RESPONSIVE IMAGES ============ */
.responsive-img img {
    width: 100%; height: auto; border-radius: 16px;
    border: 1px solid rgba(45, 212, 191, 0.28);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

/* ============ MISC ============ */
.checklist { padding-left: 1.1rem; } .checklist li { margin: 0.35rem 0; line-height: 1.5; }
.stSpinner > div { border-top-color: #14b8a6 !important; }
hr.soft { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(45,212,191,0.4), transparent); margin: 1.4rem 0; }

/* ============ RESPONSIVE — MOBILE & TABLET ============ */
@media (max-width: 900px) {
    .block-container { padding-top: 5.5rem; }
    .topnav { padding: 0.6rem 1rem; }
    .topnav .links {
        display: none; position: absolute; top: 100%; left: 0; right: 0;
        flex-direction: column; gap: 0; padding: 0.5rem 1rem 1rem 1rem;
        background: rgba(7, 16, 24, 0.96);
        backdrop-filter: blur(18px); border-bottom: 1px solid rgba(45,212,191,0.25);
        animation: fadeUp 0.35s ease both;
    }
    #nav-toggle:checked ~ .links { display: flex; }
    .topnav .links a { padding: 0.75rem 1rem; border-radius: 12px; font-size: 1rem; }
    .topnav .menu-btn { display: block; cursor: pointer; }
    .glass { padding: 1.1rem; border-radius: 16px; }
}
@media (max-width: 480px) {
    .block-container { padding-top: 5rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    .hero { padding: 1.5rem 0.8rem; }
    .stButton > button { padding: 0.85rem 1rem; font-size: 1rem; }  /* bigger touch target */
}
</style>

<!-- ambient floating orbs -->
<div class="orb orb-1"></div><div class="orb orb-2"></div><div class="orb orb-3"></div>

<!-- fixed top navbar (never hides the page) — CSS-only mobile menu -->
<nav class="topnav">
    <a class="brand" href="#home"><span class="dot"></span>🔬 Skin Lesion Analyzer</a>
    <input type="checkbox" id="nav-toggle" style="display:none;">
    <div class="links">
        <a href="#home">Home</a>
        <a href="#analyzer">Analyzer</a>
        <a href="#settings">Settings</a>
        <a href="#results">Results</a>
        <a href="#about">About</a>
    </div>
    <label for="nav-toggle" class="menu-btn">☰</label>
</nav>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. SETTINGS PANEL (moved out of the hidden sidebar → always visible)
# ----------------------------------------------------------------------------

st.markdown('<div id="settings"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass fade-up-2"><h4 style="margin:0 0 0.6rem 0;">⚙️ Decision Threshold</h4>', unsafe_allow_html=True)
c_set1, c_set2 = st.columns([3, 1])
with c_set1:
    threshold = st.slider(
        "Malignant probability above this value → classified as Malignant",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        label_visibility="collapsed",
    )
with c_set2:
    st.metric("Threshold", f"{threshold:.2f}")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 4. HERO HEADER  (anchor: #home)
# ----------------------------------------------------------------------------

st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="glass hero fade-up">
        <h1 class="grad-text float-anim">🔬 Skin Lesion Analyzer</h1>
        <p class="sub">AI-assisted dermoscopic image classification — Benign vs. Malignant</p>
        <span class="badge-disclaimer">⚕️ For research &amp; educational use only — not a medical diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model once per session
model = load_model()

# ----------------------------------------------------------------------------
# 5. UPLOAD + PREVIEW  (anchor: #analyzer)
# ----------------------------------------------------------------------------

st.markdown('<div id="analyzer"></div>', unsafe_allow_html=True)
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="glass fade-up-1">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Lesion Image")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse — .jpg / .jpeg / .png",
        type=["jpg", "jpeg", "png"], label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="glass fade-up-2" style="margin-top:1rem;">', unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze Lesion", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        analyze_clicked = False

with col_preview:
    st.markdown('<div class="glass fade-up-2 responsive-img">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded lesion image", use_container_width=True)
        st.caption(f"Original {image.size[0]} × {image.size[1]} px → resized to 224 × 224")
    else:
        st.info("Upload an image to see a live preview here.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. RESULTS  (anchor: #results)
# ----------------------------------------------------------------------------

st.markdown('<div id="results"></div>', unsafe_allow_html=True)

if uploaded_file is not None and analyze_clicked:
    with st.spinner("⚙️ Running deep-learning inference..."):
        time.sleep(0.4)
        img_batch = preprocess_image(image)
        label, confidence, probs = predict(model, img_batch)
        benign_p, malignant_p = float(probs[0]), float(probs[1])

    if malignant_p >= threshold:
        final_label, badge_class, icon = "Malignant", "badge-malignant", "🚨"
    elif benign_p >= threshold:
        final_label, badge_class, icon = "Benign", "badge-benign", "✅"
    else:
        final_label, badge_class, icon = "Uncertain — consult a dermatologist", "badge-uncertain", "⚠️"

    st.markdown('<div class="glass fade-up">', unsafe_allow_html=True)
    st.markdown("## 🧾 Analysis Results")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f'<span class="badge {badge_class}">{icon} {final_label.upper()}</span>', unsafe_allow_html=True)
    with c2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with c3:
        st.metric("Threshold", f"{threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass fade-up-2" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown("### 📊 Probability Breakdown")

    b_color = "#22c55e" if benign_p >= malignant_p else "#475569"
    m_color = "#ef4444" if malignant_p > benign_p else "#475569"

    st.markdown(f"**🟢 Benign — {benign_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{benign_p * 100:.1f}%; background:{b_color}; box-shadow:0 0 12px {b_color};"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**🔴 Malignant — {malignant_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{malignant_p * 100:.1f}%; background:{m_color}; box-shadow:0 0 12px {m_color};"></div></div>',
        unsafe_allow_html=True,
    )

    if final_label.startswith("Malignant"):
        st.warning("🚨 Features consistent with **malignancy** detected. Please consult a dermatologist promptly.")
    elif final_label.startswith("Uncertain"):
        st.info("⚠️ The model is uncertain about this lesion. Consider professional evaluation.")
    else:
        st.success("✅ This lesion appears **benign**. Continue routine skin monitoring.")
    st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is not None:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; opacity:0.8;">'
        "👆 Press <b>Analyze Lesion</b> to run the classifier.</p></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; opacity:0.8;">'
        "📁 Upload a dermoscopic image to begin analysis.</p></div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# 7. ABOUT + GUIDELINES + FOOTER  (anchor: #about)
# ----------------------------------------------------------------------------

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
c_a, c_b = st.columns(2, gap="large")

with c_a:
    st.markdown('<div class="glass fade-up-1">', unsafe_allow_html=True)
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

with c_b:
    st.markdown('<div class="glass fade-up-2">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About & Privacy")
    st.markdown(
        """
        <ul class="checklist">
            <li>🧠 Deep-learning classifier (224×224 input, TensorFlow)</li>
            <li>⚡ Model auto-downloads once, then inference runs instantly</li>
            <li>🎚️ Adjustable decision threshold for sensitivity control</li>
            <li>🔒 Images are processed in-memory only — never stored</li>
            <li>📱 Fully responsive — works on phone, tablet &amp; desktop</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="glass fade-up-3" style="margin-top:1.6rem; text-align:center;">
        <small>
        ⚕️ <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
        It is <b>not</b> a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for any skin concerns.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
