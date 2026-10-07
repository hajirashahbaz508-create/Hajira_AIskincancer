"""
Skin Lesion Analyzer — Skin Cancer Detection Web App  (v3.0 · Light Galaxy)
===========================================================================
• Clean LIGHT glassmorphism theme — white/sky surfaces, teal accents
• No emojis, no doodles — professional clinical look
• Galaxy-style animations: twinkling starfield, pastel nebula clouds,
  shooting star, staggered fade-up reveals, shimmer buttons
• Fully responsive: fixed glass navbar, CSS-only mobile menu, touch targets
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

CLASS_LABELS = ["benign", "malignant"]
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.keras"
GDRIVE_FILE_ID = "16tlVBG4-pbqmnWCUEaHCxtNCmQ_Y5lnS"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

st.set_page_config(
    page_title="Skin Lesion Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights — please wait..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model, img_batch):
    probs = model.predict(img_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_LABELS[idx], float(probs[idx]), probs


# ----------------------------------------------------------------------------
# 2. LIGHT GALAXY THEME CSS
# ----------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

/* ============ LIGHT DESIGN TOKENS ============ */
/*  bg      : airy sky gradient #f4f9fc → #e8f3f8
    ink     : #0f2a33 (deep teal-slate)
    primary : #0d9488 (teal-600)   accent : #0891b2 (cyan-700)
    glass   : rgba(255, 255, 255, 0.55–0.72)
    nebula  : pastel teal / violet / rose                                 */

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(ellipse 90% 60% at 15% 0%, rgba(165, 243, 252, 0.35) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 90% 20%, rgba(196, 181, 253, 0.28) 0%, transparent 50%),
        linear-gradient(165deg, #f6fafc 0%, #eef6fa 50%, #e6f1f7 100%);
    color: #0f2a33;
    scroll-behavior: smooth;
}
[data-testid="stAppViewContainer"] { background-attachment: fixed; }
.block-container { max-width: 1200px; padding-top: 6.5rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif; color: #0b3b45; }
p, li, span, label, small { color: #28505c; }

/* ============ GALAXY STARFIELD (light, subtle) ============ */
.stars {
    position: fixed; top: 0; left: 0; width: 3px; height: 3px;
    border-radius: 50%; background: transparent; z-index: 0; pointer-events: none;
    box-shadow:
        90vw 12vh #67e8f9,  20vw 34vh #99f6e4,  55vw 8vh  #c4b5fd,  75vw 48vh #67e8f9,
        10vw 62vh #99f6e4,  35vw 78vh #c4b5fd,  85vw 70vh #67e8f9,  48vw 55vh #99f6e4,
        5vw  18vh #c4b5fd,  65vw 88vh #67e8f9,  28vw 5vh  #99f6e4,  95vw 30vh #c4b5fd,
        15vw 90vh #67e8f9,  60vw 25vh #99f6e4,  40vw 40vh #c4b5fd,  80vw 60vh #99f6e4;
    animation: twinkle 3.5s ease-in-out infinite;
}
.stars-2 {
    position: fixed; top: 0; left: 0; width: 2px; height: 2px;
    border-radius: 50%; background: transparent; z-index: 0; pointer-events: none;
    box-shadow:
        12vw 8vh  #5eead4,  45vw 20vh #a5b4fc,  70vw 15vh #5eead4,  25vw 50vh #a5b4fc,
        90vw 42vh #5eead4,  55vw 65vh #a5b4fc,  8vw  75vh  #5eead4,  33vw 30vh #a5b4fc,
        77vw 82vh #5eead4,  18vw 45vh #a5b4fc,  62vw 38vh #5eead4,  42vw 92vh #a5b4fc;
    animation: twinkle 5s ease-in-out -2s infinite;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.9); }
    50%      { opacity: 1;    transform: scale(1.15); }
}

/* shooting star */
.shooting-star {
    position: fixed; top: 12vh; right: -10vw; width: 120px; height: 2px; z-index: 0;
    background: linear-gradient(90deg, rgba(13,148,136,0.9), transparent);
    border-radius: 999px; pointer-events: none;
    animation: shoot 7s linear infinite;
}
.shooting-star::after {
    content: ""; position: absolute; right: 0; top: -2px;
    width: 6px; height: 6px; border-radius: 50%;
    background: #0d9488; box-shadow: 0 0 12px 3px rgba(13,148,136,0.6);
}
@keyframes shoot {
    0%   { transform: translate(0, 0) rotate(-25deg); opacity: 0; }
    3%   { opacity: 1; }
    14%  { transform: translate(-120vw, 55vh) rotate(-25deg); opacity: 0; }
    100% { transform: translate(-120vw, 55vh) rotate(-25deg); opacity: 0; }
}

/* pastel nebula clouds */
.nebula {
    position: fixed; border-radius: 50%; filter: blur(100px);
    pointer-events: none; z-index: 0;
    animation: drift 26s ease-in-out infinite alternate;
}
.neb-1 { width: 480px; height: 480px; top: -140px; left: -140px; opacity: 0.5; background: radial-gradient(circle, #99f6e4 0%, transparent 70%); }
.neb-2 { width: 420px; height: 420px; bottom: -120px; right: -100px; opacity: 0.4; background: radial-gradient(circle, #c4b5fd 0%, transparent 70%); animation-delay: -9s; }
.neb-3 { width: 320px; height: 320px; top: 45%; left: 50%; opacity: 0.3; background: radial-gradient(circle, #fda4af 0%, transparent 70%); animation-delay: -16s; }
@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    50%  { transform: translate(70px, 45px) scale(1.12); }
    100% { transform: translate(-50px, 80px) scale(0.94); }
}

/* ============ FIXED GLASS TOP NAVBAR ============ */
.topnav {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 2rem;
    background: rgba(255, 255, 255, 0.68);
    backdrop-filter: blur(18px) saturate(1.4);
    -webkit-backdrop-filter: blur(18px) saturate(1.4);
    border-bottom: 1px solid rgba(13, 148, 136, 0.18);
    box-shadow: 0 6px 24px rgba(13, 60, 70, 0.10);
    animation: slideDown 0.6s ease both;
}
@keyframes slideDown { from { transform: translateY(-100%); } to { transform: translateY(0); } }
.topnav .brand {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem;
    color: #0b3b45; text-decoration: none; letter-spacing: 0.2px;
    display: flex; align-items: center; gap: 0.55rem; white-space: nowrap;
}
.topnav .brand .pulse-dot {
    width: 10px; height: 10px; border-radius: 50%; background: #0d9488;
    box-shadow: 0 0 0 0 rgba(13, 148, 136, 0.5);
    animation: pulseRing 2.2s ease-out infinite;
}
@keyframes pulseRing {
    0%   { box-shadow: 0 0 0 0 rgba(13,148,136,0.45); }
    70%  { box-shadow: 0 0 0 12px rgba(13,148,136,0); }
    100% { box-shadow: 0 0 0 0 rgba(13,148,136,0); }
}
.topnav .links { display: flex; gap: 1.2rem; }
.topnav .links a {
    color: #28505c; text-decoration: none; font-weight: 600; font-size: 0.88rem;
    padding: 0.35rem 0.95rem; border-radius: 999px;
    border: 1px solid transparent; transition: all 0.3s ease; white-space: nowrap;
}
.topnav .links a:hover {
    color: #ffffff; background: linear-gradient(120deg, #0d9488, #0891b2);
    box-shadow: 0 4px 16px rgba(13, 148, 136, 0.35);
    transform: translateY(-1px);
}
.topnav .menu-btn {
    display: none; background: rgba(255,255,255,0.6);
    border: 1px solid rgba(13,148,136,0.35); color: #0b3b45;
    border-radius: 10px; font-size: 1.15rem; padding: 0.25rem 0.75rem;
    cursor: pointer; transition: all 0.3s ease;
}
.topnav .menu-btn:hover { background: rgba(13,148,136,0.12); }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="collapsedControl"], [data-testid="stSidebar"] { display: none !important; }

/* ============ GLASS CARDS (light) ============ */
.glass {
    position: relative; z-index: 1;
    background: rgba(255, 255, 255, 0.58);
    backdrop-filter: blur(16px) saturate(1.3);
    -webkit-backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(13, 148, 136, 0.18);
    border-radius: 20px;
    box-shadow: 0 8px 30px rgba(13, 60, 70, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.9);
    padding: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.glass:hover {
    transform: translateY(-4px);
    border-color: rgba(13, 148, 136, 0.4);
    box-shadow: 0 16px 44px rgba(13, 148, 136, 0.16), inset 0 1px 0 rgba(255,255,255,1);
}

/* ============ ANIMATION CLASSES ============ */
@keyframes fadeUp { from { opacity: 0; transform: translateY(26px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn  { 0% { opacity: 0; transform: scale(0.85); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes floatY { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes gradientMove { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

.fade-up   { animation: fadeUp 0.7s ease both; }
.fade-up-1 { animation: fadeUp 0.7s ease 0.12s both; }
.fade-up-2 { animation: fadeUp 0.7s ease 0.24s both; }
.fade-up-3 { animation: fadeUp 0.7s ease 0.36s both; }
.pop-in    { animation: popIn 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
.float-anim { animation: floatY 4.5s ease-in-out infinite; }

.grad-text {
    background: linear-gradient(90deg, #0d9488, #0891b2, #0ea5e9, #0d9488);
    background-size: 300% auto;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 6s linear infinite;
}

/* ============ HERO ============ */
.hero { text-align: center; padding: 2.4rem 1.5rem 2rem 1.5rem; margin-bottom: 1.6rem; overflow: hidden; }
.hero h1 { font-size: clamp(1.9rem, 5vw, 3rem); font-weight: 800; letter-spacing: -1px; margin: 0; }
.hero p.sub { color: #3a6470; font-weight: 300; margin: 0.5rem 0 0 0; font-size: clamp(0.9rem, 2.5vw, 1.1rem); }
.hero .disclaimer-pill {
    display: inline-block; margin-top: 1rem; padding: 0.4rem 1.15rem; border-radius: 999px;
    background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(217, 119, 6, 0.4);
    color: #92400e; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.4px;
    animation: popIn 0.7s ease 0.4s both;
}

/* ============ RESULT BADGES ============ */
.badge {
    display: inline-block; padding: 0.6rem 1.7rem; border-radius: 999px;
    font-size: clamp(0.95rem, 3vw, 1.4rem); font-weight: 800; letter-spacing: 1.5px;
    animation: popIn 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.badge-benign    { background: rgba(34, 197, 94, 0.14); border: 1px solid rgba(22,163,74,0.5);  color: #15803d; box-shadow: 0 4px 18px rgba(34,197,94,0.25); }
.badge-malignant { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(220,38,38,0.5);  color: #b91c1c; box-shadow: 0 4px 18px rgba(239,68,68,0.25); }
.badge-uncertain { background: rgba(245, 158, 11, 0.13); border: 1px solid rgba(217,119,6,0.5); color: #92400e; box-shadow: 0 4px 18px rgba(245,158,11,0.25); }

/* ============ PROBABILITY BARS ============ */
.prob-track {
    width: 100%; height: 16px; border-radius: 999px;
    background: rgba(13, 148, 136, 0.08); overflow: hidden; margin: 0.4rem 0 1.1rem 0;
    border: 1px solid rgba(13, 148, 136, 0.12);
}
.prob-fill {
    height: 100%; border-radius: 999px; width: 0;
    transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
    animation: fadeUp 0.5s ease both;
}

/* ============ UPLOADER ============ */
[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(12px);
    border: 1.5px dashed rgba(13, 148, 136, 0.4);
    border-radius: 18px; transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    transform: scale(1.015); border-color: #0891b2;
    box-shadow: 0 0 26px rgba(8, 145, 178, 0.25);
}

/* ============ BUTTONS (shimmer) ============ */
.stButton > button {
    width: 100%; border-radius: 14px; border: 1px solid rgba(13,148,136,0.35);
    background: linear-gradient(120deg, #14b8a6, #0891b2, #14b8a6);
    background-size: 200% auto;
    color: #ffffff; font-weight: 700; padding: 0.7rem 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 24px rgba(13, 148, 136, 0.4);
    animation: shimmer 2s linear infinite;
}

/* ============ METRICS / SLIDER / SPINNER ============ */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.55); backdrop-filter: blur(10px);
    border: 1px solid rgba(13, 148, 136, 0.18); border-radius: 16px;
    padding: 0.9rem 1.1rem; transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(13, 60, 70, 0.06);
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(13,148,136,0.4); }
[data-testid="stMetricValue"] { color: #0d9488 !important; }
[data-testid="stMetricLabel"] { color: #3a6470 !important; }
.stSlider label { color: #28505c !important; }
.stSpinner > div { border-top-color: #0d9488 !important; }

/* ============ RESPONSIVE IMAGES ============ */
.responsive-img img {
    width: 100%; height: auto; border-radius: 16px;
    border: 1px solid rgba(13, 148, 136, 0.25);
    box-shadow: 0 8px 24px rgba(13, 60, 70, 0.14);
}

/* ============ MISC ============ */
.checklist { padding-left: 1.1rem; } .checklist li { margin: 0.4rem 0; line-height: 1.55; color: #28505c; }
hr.soft { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(13,148,136,0.35), transparent); margin: 1.4rem 0; }
.stInfo, .stWarning, .stSuccess { border-radius: 14px !important; }

/* ============ RESPONSIVE — TABLET & MOBILE ============ */
@media (max-width: 900px) {
    .block-container { padding-top: 5.5rem; }
    .topnav { padding: 0.6rem 1rem; }
    .topnav .links {
        display: none; position: absolute; top: 100%; left: 0; right: 0;
        flex-direction: column; gap: 0; padding: 0.5rem 1rem 1rem 1rem;
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(20px); border-bottom: 1px solid rgba(13,148,136,0.2);
        box-shadow: 0 12px 30px rgba(13, 60, 70, 0.12);
        animation: fadeUp 0.35s ease both;
    }
    #nav-toggle:checked ~ .links { display: flex; }
    .topnav .links a { padding: 0.8rem 1rem; border-radius: 12px; font-size: 1rem; }
    .topnav .menu-btn { display: block; }
    .glass { padding: 1.1rem; border-radius: 16px; }
}
@media (max-width: 480px) {
    .block-container { padding-top: 5rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    .hero { padding: 1.6rem 0.8rem; }
    .stButton > button { padding: 0.9rem 1rem; font-size: 1rem; }
}
</style>

<!-- galaxy: starfield + shooting star + nebula clouds -->
<div class="stars"></div>
<div class="stars-2"></div>
<div class="shooting-star"></div>
<div class="nebula neb-1"></div><div class="nebula neb-2"></div><div class="nebula neb-3"></div>

<!-- fixed top navbar — CSS-only mobile menu -->
<nav class="topnav">
    <a class="brand" href="#home"><span class="pulse-dot"></span>Skin Lesion Analyzer</a>
    <input type="checkbox" id="nav-toggle" style="display:none;">
    <div class="links">
        <a href="#home">Home</a>
        <a href="#analyzer">Analyzer</a>
        <a href="#settings">Settings</a>
        <a href="#results">Results</a>
        <a href="#about">About</a>
    </div>
    <label for="nav-toggle" class="menu-btn">&#9776;</label>
</nav>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 3. SETTINGS PANEL (always visible on main page)
# ----------------------------------------------------------------------------

st.markdown('<div id="settings"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass fade-up-2"><h4 style="margin:0 0 0.6rem 0;">Decision Threshold</h4>', unsafe_allow_html=True)
c_set1, c_set2 = st.columns([3, 1])
with c_set1:
    threshold = st.slider(
        "Malignant probability above this value is classified as Malignant",
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
        <h1 class="grad-text float-anim">Skin Lesion Analyzer</h1>
        <p class="sub">AI-assisted dermoscopic image classification — Benign vs. Malignant</p>
        <span class="disclaimer-pill">For research and educational use only — not a medical diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

# ----------------------------------------------------------------------------
# 5. UPLOAD + PREVIEW  (anchor: #analyzer)
# ----------------------------------------------------------------------------

st.markdown('<div id="analyzer"></div>', unsafe_allow_html=True)
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="glass fade-up-1">', unsafe_allow_html=True)
    st.markdown("### Upload Lesion Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse — accepted formats: .jpg, .jpeg, .png",
        type=["jpg", "jpeg", "png"], label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="glass fade-up-2" style="margin-top:1rem;">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Lesion", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        analyze_clicked = False

with col_preview:
    st.markdown('<div class="glass fade-up-2 responsive-img">', unsafe_allow_html=True)
    st.markdown("### Image Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded lesion image", use_container_width=True)
        st.caption(f"Original {image.size[0]} x {image.size[1]} px — resized to 224 x 224 for inference")
    else:
        st.info("Upload an image to see a live preview here.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. RESULTS  (anchor: #results)
# ----------------------------------------------------------------------------

st.markdown('<div id="results"></div>', unsafe_allow_html=True)

if uploaded_file is not None and analyze_clicked:
    with st.spinner("Running deep-learning inference..."):
        time.sleep(0.4)
        img_batch = preprocess_image(image)
        label, confidence, probs = predict(model, img_batch)
        benign_p, malignant_p = float(probs[0]), float(probs[1])

    if malignant_p >= threshold:
        final_label, badge_class = "MALIGNANT", "badge-malignant"
    elif benign_p >= threshold:
        final_label, badge_class = "BENIGN", "badge-benign"
    else:
        final_label, badge_class = "UNCERTAIN — CONSULT A DERMATOLOGIST", "badge-uncertain"

    st.markdown('<div class="glass fade-up">', unsafe_allow_html=True)
    st.markdown("## Analysis Results")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f'<span class="badge {badge_class}">{final_label}</span>', unsafe_allow_html=True)
    with c2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with c3:
        st.metric("Threshold", f"{threshold:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass fade-up-2" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown("### Probability Breakdown")

    b_color = "#22c55e" if benign_p >= malignant_p else "#94a3b8"
    m_color = "#ef4444" if malignant_p > benign_p else "#94a3b8"

    st.markdown(f"**Benign — {benign_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{benign_p * 100:.1f}%; background:{b_color};"></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Malignant — {malignant_p * 100:.1f}%**")
    st.markdown(
        f'<div class="prob-track"><div class="prob-fill" style="width:{malignant_p * 100:.1f}%; background:{m_color};"></div></div>',
        unsafe_allow_html=True,
    )

    if final_label.startswith("MALIGNANT"):
        st.warning("Features consistent with malignancy were detected. Please consult a dermatologist promptly.")
    elif final_label.startswith("UNCERTAIN"):
        st.info("The model is uncertain about this lesion. Consider professional evaluation.")
    else:
        st.success("This lesion appears benign. Continue routine skin monitoring.")
    st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is not None:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; color:#3a6470;">'
        "Press <b>Analyze Lesion</b> to run the classifier.</p></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; color:#3a6470;">'
        "Upload a dermoscopic image to begin analysis.</p></div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# 7. ABOUT + GUIDELINES + FOOTER  (anchor: #about)
# ----------------------------------------------------------------------------

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
c_a, c_b = st.columns(2, gap="large")

with c_a:
    st.markdown('<div class="glass fade-up-1">', unsafe_allow_html=True)
    st.markdown("### Image Guidelines")
    st.markdown(
        """
        <ul class="checklist">
            <li>Use a <b>clear, in-focus</b> close-up of the lesion</li>
            <li>Even, bright lighting — avoid harsh shadows</li>
            <li>Fill most of the frame with the lesion</li>
            <li>Accepted formats: <b>.jpg</b>, <b>.jpeg</b>, or <b>.png</b></li>
            <li>Avoid blurry, filtered, or heavily compressed images</li>
            <li>Hair, rulers, or ink marks may reduce accuracy</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c_b:
    st.markdown('<div class="glass fade-up-2">', unsafe_allow_html=True)
    st.markdown("### About and Privacy")
    st.markdown(
        """
        <ul class="checklist">
            <li>Deep-learning classifier with a 224 x 224 input, built on TensorFlow</li>
            <li>Model weights auto-download once, then inference runs instantly</li>
            <li>Adjustable decision threshold for sensitivity control</li>
            <li>Images are processed in memory only — never stored</li>
            <li>Fully responsive — works on phone, tablet, and desktop</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="glass fade-up-3" style="margin-top:1.6rem; text-align:center;">
        <small style="color:#3a6470;">
        <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for any skin concerns.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
