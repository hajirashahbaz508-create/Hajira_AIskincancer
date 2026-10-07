import os
import time
import gdown
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# 1. CORE CONFIGURATION & BACKEND  (unchanged)
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
# 2. THEME CSS  (light galaxy + glass + entrance / result animations)
# ----------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap');

/* ============ DESIGN TOKENS ============
   ink      #0b3b45   body text #28505c   teal #0d9488   cyan #0891b2
   violet   #7c3aed   emerald  #10b981   alert #ef4444   amber #f59e0b
   ======================================= */

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(ellipse 90% 60% at 15% 0%, rgba(165, 243, 252, 0.38) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 90% 20%, rgba(196, 181, 253, 0.30) 0%, transparent 50%),
        linear-gradient(165deg, #f6fafc 0%, #eef6fa 50%, #e6f1f7 100%);
    color: #0f2a33;
    scroll-behavior: smooth;
}
[data-testid="stAppViewContainer"] { background-attachment: fixed; }
.block-container { max-width: 1200px; padding-top: 6.5rem; padding-bottom: 3rem; position: relative; z-index: 1; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif; color: #0b3b45; }
p, li, span, label, small { color: #28505c; }

/* ============ GALAXY BACKDROP ============ */
.stars {
    position: fixed; top: 0; left: 0; width: 3px; height: 3px;
    border-radius: 50%; background: transparent; z-index: 0; pointer-events: none;
    box-shadow:
        90vw 12vh #67e8f9, 20vw 34vh #99f6e4, 55vw 8vh #c4b5fd, 75vw 48vh #67e8f9,
        10vw 62vh #99f6e4, 35vw 78vh #c4b5fd, 85vw 70vh #67e8f9, 48vw 55vh #99f6e4,
        5vw 18vh #c4b5fd, 65vw 88vh #67e8f9, 28vw 5vh #99f6e4, 95vw 30vh #c4b5fd,
        15vw 90vh #67e8f9, 60vw 25vh #99f6e4, 40vw 40vh #c4b5fd, 80vw 60vh #99f6e4;
    animation: twinkle 3.5s ease-in-out infinite;
}
.stars-2 {
    position: fixed; top: 0; left: 0; width: 2px; height: 2px;
    border-radius: 50%; background: transparent; z-index: 0; pointer-events: none;
    box-shadow:
        12vw 8vh #5eead4, 45vw 20vh #a5b4fc, 70vw 15vh #5eead4, 25vw 50vh #a5b4fc,
        90vw 42vh #5eead4, 55vw 65vh #a5b4fc, 8vw 75vh #5eead4, 33vw 30vh #a5b4fc,
        77vw 82vh #5eead4, 18vw 45vh #a5b4fc, 62vw 38vh #5eead4, 42vw 92vh #a5b4fc;
    animation: twinkle 5s ease-in-out -2s infinite;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.9); }
    50%      { opacity: 1;    transform: scale(1.15); }
}
.shooting-star {
    position: fixed; top: 12vh; right: -12vw; width: 190px; height: 5px; z-index: 0;
    background: linear-gradient(90deg, rgba(13,148,136,0.95), rgba(34,211,238,0.55), transparent);
    border-radius: 999px; pointer-events: none;
    box-shadow: 0 0 14px rgba(13, 148, 136, 0.5);
    animation: shoot 7s linear infinite;
}
.shooting-star::after {
    content: ""; position: absolute; right: -2px; top: -3.5px;
    width: 12px; height: 12px; border-radius: 50%;
    background: #0d9488; box-shadow: 0 0 20px 6px rgba(13,148,136,0.7);
}
@keyframes shoot {
    0%   { transform: translate(0, 0) rotate(-25deg); opacity: 0; }
    3%   { opacity: 1; }
    14%  { transform: translate(-120vw, 55vh) rotate(-25deg); opacity: 0; }
    100% { transform: translate(-120vw, 55vh) rotate(-25deg); opacity: 0; }
}
.nebula { position: fixed; border-radius: 50%; filter: blur(100px); pointer-events: none; z-index: 0; animation: drift 26s ease-in-out infinite alternate; }
.neb-1 { width: 480px; height: 480px; top: -140px; left: -140px; opacity: 0.5; background: radial-gradient(circle, #99f6e4 0%, transparent 70%); }
.neb-2 { width: 420px; height: 420px; bottom: -120px; right: -100px; opacity: 0.4; background: radial-gradient(circle, #c4b5fd 0%, transparent 70%); animation-delay: -9s; }
.neb-3 { width: 320px; height: 320px; top: 45%; left: 50%; opacity: 0.3; background: radial-gradient(circle, #fda4af 0%, transparent 70%); animation-delay: -16s; }
@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    50%  { transform: translate(70px, 45px) scale(1.12); }
    100% { transform: translate(-50px, 80px) scale(0.94); }
}

/* ============ TOP NAVBAR ============ */
#home, #analyzer, #settings, #results, #about { scroll-margin-top: 90px; }
.topnav {
    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 0.7rem clamp(1rem, 3vw, 2rem);
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(18px) saturate(1.4); -webkit-backdrop-filter: blur(18px) saturate(1.4);
    border-bottom: 1px solid rgba(13, 148, 136, 0.18);
    box-shadow: 0 6px 24px rgba(13, 60, 70, 0.08);
    animation: navDrop 0.7s cubic-bezier(0.22, 1, 0.36, 1) backwards;
}
@keyframes navDrop { from { transform: translateY(-100%); } to { transform: translateY(0); } }
.topnav .brand {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem;
    color: #0b3b45; text-decoration: none; display: flex; align-items: center; gap: 0.55rem; white-space: nowrap;
}
.topnav .brand .pulse-dot { width: 10px; height: 10px; border-radius: 50%; background: #0d9488; animation: pulseRing 2.2s ease-out infinite; }
@keyframes pulseRing {
    0%   { box-shadow: 0 0 0 0 rgba(13,148,136,0.45); }
    70%  { box-shadow: 0 0 0 12px rgba(13,148,136,0); }
    100% { box-shadow: 0 0 0 0 rgba(13,148,136,0); }
}
.topnav .links { display: flex; gap: 0.3rem; }
.topnav a.nav-link {
    color: #28505c; text-decoration: none; font-weight: 600; font-size: 0.88rem;
    padding: 0.4rem 0.95rem; border-radius: 999px; transition: all 0.3s ease; white-space: nowrap;
}
.topnav a.nav-link:hover {
    color: #fff; background: linear-gradient(120deg, #7c3aed, #0d9488);
    box-shadow: 0 4px 16px rgba(13, 148, 136, 0.3);
}
/* mobile menu: <details> needs no JavaScript */
.mobile-menu { display: none; position: relative; }
.mobile-menu summary {
    list-style: none; cursor: pointer; width: 42px; height: 42px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem; color: #0b3b45;
    background: rgba(255,255,255,0.8); border: 1px solid rgba(13,148,136,0.35);
}
.mobile-menu summary::-webkit-details-marker { display: none; }
.mobile-menu .drop {
    position: absolute; right: 0; top: calc(100% + 0.7rem); min-width: 210px;
    display: flex; flex-direction: column; padding: 0.5rem; border-radius: 18px;
    background: rgba(255,255,255,0.97); border: 1px solid rgba(13,148,136,0.2);
    box-shadow: 0 18px 44px rgba(13, 60, 70, 0.18); animation: dropIn 0.3s ease both;
}
.mobile-menu .drop a { padding: 0.8rem 1rem; border-radius: 12px; font-size: 1rem; }
@keyframes dropIn { from { opacity: 0; transform: translateY(-10px) scale(0.96); } to { opacity: 1; transform: none; } }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="collapsedControl"], [data-testid="stSidebar"] { display: none !important; }

/* ============ ENTRANCE: slideDownFade ============
   Float down, stay gently suspended (raised + soft glow) for ~3.6s,
   then settle into the static glass layout. Cards cascade 0.2s apart.   */
@keyframes slideDownFade {
    0%   { opacity: 0; transform: translateY(-56px) scale(0.98); }
    12%  { opacity: 1; transform: translateY(-10px) scale(1.005);
           box-shadow: 0 24px 56px rgba(124,58,237,0.20), 0 0 0 2px rgba(13,148,136,0.22); }
    85%  { opacity: 1; transform: translateY(-10px) scale(1.005);
           box-shadow: 0 24px 56px rgba(124,58,237,0.20), 0 0 0 2px rgba(13,148,136,0.22); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
.intro-1 { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.0s backwards; }
.intro-2 { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.2s backwards; }
.intro-3 { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.4s backwards; }
.intro-4 { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.6s backwards; }

/* scroll reveal for lower sections (falls back to a simple fade-up) */
@keyframes revealScroll { from { opacity: 0; transform: translateY(44px) scale(0.97); } to { opacity: 1; transform: none; } }
.reveal { animation: fadeUp 0.8s ease both; }
@supports (animation-timeline: view()) {
    .reveal { animation: revealScroll linear both; animation-timeline: view(); animation-range: entry 0% entry 55%; }
}

/* hero chips */
.chips { display: flex; flex-wrap: wrap; gap: 0.55rem; justify-content: center; margin-top: 1.2rem; }
.chip {
    padding: 0.35rem 0.95rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    color: #4c2a9a; background: rgba(255,255,255,0.8); border: 1px solid rgba(124,58,237,0.25);
    animation: chipFloat 5s ease-in-out infinite;
}
.chip:nth-child(2) { animation-delay: -1.6s; } .chip:nth-child(3) { animation-delay: -3.2s; }
@keyframes chipFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }

/* ============ PINTEREST-STYLE GLASS FRAMES ============ */
.glass,
.st-key-upload_card, .st-key-preview_card, .st-key-settings_card, .st-key-analyze_card {
    position: relative;
    background: rgba(255, 255, 255, 0.60);
    backdrop-filter: blur(16px) saturate(1.3); -webkit-backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid rgba(13, 148, 136, 0.16);
    border-radius: 20px;
    box-shadow:
        0 2px 4px rgba(13, 60, 70, 0.05),
        0 10px 24px rgba(13, 60, 70, 0.08),
        0 28px 60px rgba(13, 148, 136, 0.10),
        inset 0 1px 0 rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.35s ease, border-color 0.35s ease;
}
.glass:hover,
.st-key-upload_card:hover, .st-key-preview_card:hover, .st-key-settings_card:hover, .st-key-analyze_card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: rgba(13, 148, 136, 0.4);
    box-shadow:
        0 4px 8px rgba(13, 60, 70, 0.06),
        0 18px 36px rgba(13, 60, 70, 0.12),
        0 40px 80px rgba(13, 148, 136, 0.18),
        inset 0 1px 0 #fff;
}
.st-key-upload_card { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.25s backwards; }
.st-key-preview_card { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.5s backwards; }
.st-key-analyze_card { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.6s backwards; }
.st-key-settings_card { animation: slideDownFade 3.8s cubic-bezier(0.22, 1, 0.36, 1) 0.75s backwards; }

/* ============ MISC ANIMATION HELPERS ============ */
@keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
@keyframes popIn  { 0% { opacity: 0; transform: scale(0.85); } 70% { transform: scale(1.04); } 100% { opacity: 1; transform: scale(1); } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes floatY { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes gradientMove { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes grow { from { width: 0; } to { width: var(--w); } }
.fade-up   { animation: fadeUp 0.7s ease both; }
.fade-up-1 { animation: fadeUp 0.7s ease 0.12s both; }
.fade-up-2 { animation: fadeUp 0.7s ease 0.24s both; }
.fade-up-3 { animation: fadeUp 0.7s ease 0.36s both; }
.float-anim { animation: floatY 4.5s ease-in-out infinite; }
.grad-text {
    background: linear-gradient(90deg, #0d9488, #7c3aed, #0891b2, #0d9488);
    background-size: 300% auto; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: gradientMove 6s linear infinite;
}

/* ============ HERO ============ */
.hero { text-align: center; padding: 2.6rem 1.5rem 2.2rem 1.5rem; margin-bottom: 1.6rem; overflow: hidden; }
.hero h1 { font-size: clamp(1.9rem, 5vw, 3rem); font-weight: 800; letter-spacing: -1px; margin: 0; }
.hero p.sub { color: #3a6470; font-weight: 300; margin: 0.5rem 0 0 0; font-size: clamp(0.9rem, 2.5vw, 1.1rem); }
.hero .disclaimer-pill {
    display: inline-block; margin-top: 1rem; padding: 0.4rem 1.15rem; border-radius: 999px;
    background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(217, 119, 6, 0.4);
    color: #92400e; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.4px;
}

/* ============ FILE UPLOADER (violet → teal redesign) ============ */
[data-testid="stFileUploader"] { margin-top: 0.2rem; }
[data-testid="stFileUploaderDropzone"] {
    position: relative; gap: 1rem; padding: 1.6rem 1.4rem;
    background:
        linear-gradient(135deg, rgba(124, 58, 237, 0.13) 0%, rgba(99, 102, 241, 0.10) 45%, rgba(20, 184, 166, 0.16) 100%),
        rgba(255, 255, 255, 0.72);
    border: 2px dashed rgba(124, 58, 237, 0.55);
    border-radius: 20px;
    transition: all 0.35s ease;
    overflow: hidden;
}
[data-testid="stFileUploaderDropzone"]::after {   /* soft sweeping light */
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(110deg, transparent 35%, rgba(255,255,255,0.65) 50%, transparent 65%);
    background-size: 250% 100%; animation: shimmer 4.5s linear infinite; opacity: 0.7;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0d9488;
    background:
        linear-gradient(135deg, rgba(124, 58, 237, 0.20) 0%, rgba(20, 184, 166, 0.24) 100%),
        rgba(255, 255, 255, 0.8);
    box-shadow: 0 0 0 5px rgba(124, 58, 237, 0.10), 0 14px 34px rgba(124, 58, 237, 0.22);
    transform: scale(1.01);
}
[data-testid="stFileUploaderDropzone"] svg { display: none; }
[data-testid="stFileUploaderDropzone"]::before {   /* upload badge */
    content: "\\2191"; flex: 0 0 auto; display: flex; align-items: center; justify-content: center;
    width: 54px; height: 54px; border-radius: 50%;
    font-size: 1.7rem; font-weight: 800; color: #fff;
    background: linear-gradient(135deg, #7c3aed, #0d9488);
    box-shadow: 0 8px 22px rgba(124, 58, 237, 0.4);
    animation: floatY 3.2s ease-in-out infinite;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #4c2a9a !important; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] button {
    border: none !important; border-radius: 14px !important; padding: 0.6rem 1.4rem !important;
    background: linear-gradient(120deg, #7c3aed, #0d9488) !important;
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35); transition: all 0.3s ease; position: relative; z-index: 1;
}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * { color: #fff !important; font-weight: 700; }
[data-testid="stFileUploaderDropzone"] button:hover { transform: translateY(-2px) scale(1.04); box-shadow: 0 10px 26px rgba(13, 148, 136, 0.45); }
[data-testid="stFileUploaderFile"] {
    background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 14px; padding: 0.4rem 0.7rem;
}

/* ============ BUTTON ============ */
.stButton > button {
    width: 100%; border-radius: 16px; border: 2px solid #0f766e;
    background: linear-gradient(120deg, #0d9488, #0891b2, #0d9488); background-size: 200% auto;
    font-weight: 800; font-size: 1.1rem; letter-spacing: 1.5px; text-transform: uppercase;
    padding: 0.95rem 1rem; box-shadow: 0 6px 20px rgba(13, 148, 136, 0.35); transition: all 0.3s ease;
}
.stButton > button, .stButton > button p, .stButton > button * { color: #ffffff !important; }
.stButton > button:hover { transform: translateY(-2px) scale(1.02); box-shadow: 0 10px 28px rgba(13, 148, 136, 0.45); animation: shimmer 2s linear infinite; }

/* ============ METRICS / SLIDER / SPINNER ============ */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.6); border: 1px solid rgba(13, 148, 136, 0.18);
    border-radius: 16px; padding: 0.9rem 1.1rem; box-shadow: 0 4px 16px rgba(13, 60, 70, 0.06);
}
[data-testid="stMetricValue"] { color: #0d9488 !important; }
[data-testid="stMetricLabel"] { color: #3a6470 !important; }
.stSlider label { color: #28505c !important; }
.stSpinner > div { border-top-color: #0d9488 !important; }
.responsive-img img { border-radius: 16px; border: 1px solid rgba(13, 148, 136, 0.25); box-shadow: 0 8px 24px rgba(13, 60, 70, 0.14); }
.checklist { padding-left: 1.1rem; } .checklist li { margin: 0.4rem 0; line-height: 1.55; }
hr.soft { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(13,148,136,0.35), transparent); margin: 1.6rem 0; }

/* ============ RESULT CARDS ============ */
@property --p { syntax: '<number>';  inherits: false; initial-value: 0; }
@property --n { syntax: '<integer>'; inherits: false; initial-value: 0; }
@keyframes gaugeIn { from { --p: 0; --n: 0; } to { --p: var(--target); --n: var(--tint); } }
@keyframes ripple  { 0% { transform: scale(0.92); opacity: 0.55; } 100% { transform: scale(1.5); opacity: 0; } }
@keyframes drawStroke { to { stroke-dashoffset: 0; } }
@keyframes riseStar { 0% { transform: translateY(24px) scale(0.4); opacity: 0; } 20% { opacity: 1; } 100% { transform: translateY(-230px) scale(1); opacity: 0; } }
@keyframes sweep { 0% { transform: translateX(-120%) skewX(-18deg); } 100% { transform: translateX(380%) skewX(-18deg); } }
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 2px rgba(239,68,68,0.5), 0 0 16px 2px rgba(245,158,11,0.30), 0 18px 44px rgba(239,68,68,0.16); }
    50%      { box-shadow: 0 0 0 4px rgba(239,68,68,0.85), 0 0 40px 10px rgba(245,158,11,0.50), 0 18px 54px rgba(239,68,68,0.30); }
}
@keyframes alertEnter { 0% { opacity: 0; transform: translateX(-36px); } 60% { opacity: 1; transform: translateX(6px); } 100% { opacity: 1; transform: none; } }
@keyframes shakeOnce { 0%,100% { transform: translateX(0); } 20% { transform: translateX(-7px); } 40% { transform: translateX(6px); } 60% { transform: translateX(-4px); } 80% { transform: translateX(3px); } }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes badgeGlow { 0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.45), 0 8px 22px rgba(16,185,129,0.28); } 50% { box-shadow: 0 0 0 12px rgba(16,185,129,0), 0 8px 28px rgba(16,185,129,0.42); } }

.rise { animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) calc(var(--i, 0) * 0.12s + 0.1s) both; }
.result-card { position: relative; border-radius: 24px; padding: clamp(1.1rem, 3vw, 2rem); overflow: hidden; }
.result-card > * { position: relative; z-index: 1; }
.result-main { display: grid; grid-template-columns: auto 1fr; gap: clamp(1.2rem, 4vw, 2.6rem); align-items: center; }
.result-info { min-width: 0; }
.status-row { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; }
.status-chip {
    display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0.9rem; border-radius: 999px;
    font-size: 0.82rem; font-weight: 700; color: #fff;
}
.status-chip::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #fff; animation: blink 1.4s ease-in-out infinite; }
.result-title { font-family: 'Sora', sans-serif; font-weight: 800; font-size: clamp(1.25rem, 3.4vw, 1.8rem); line-height: 1.2; margin: 0.7rem 0 0.3rem 0; }
.result-sub { margin: 0; font-size: 0.97rem; line-height: 1.55; }

/* animated gauge: ring fills and number counts up */
.gauge-wrap { position: relative; display: grid; place-items: center; padding: 14px; }
.gauge {
    --size: clamp(150px, 22vw, 210px);
    width: var(--size); aspect-ratio: 1; border-radius: 50%; position: relative; display: grid; place-items: center;
    background: conic-gradient(var(--g1), var(--g2) calc(var(--p) * 3.6deg), rgba(15,42,51,0.09) 0);
    animation: gaugeIn 1.9s cubic-bezier(0.22, 1, 0.36, 1) 0.35s both;
    box-shadow: 0 14px 34px rgba(13, 60, 70, 0.14);
}
.gauge::before { content: ""; position: absolute; inset: 15px; border-radius: 50%; background: #fff; box-shadow: inset 0 2px 8px rgba(13,60,70,0.08); }
.gauge-center { position: relative; z-index: 1; text-align: center; line-height: 1.05; }
.gauge .num { font-family: 'Sora', sans-serif; font-weight: 800; font-size: clamp(2rem, 6vw, 2.7rem); counter-reset: n var(--n); color: var(--g2); }
.gauge .num::after { content: counter(n) "%"; }
.gauge .cap { display: block; margin-top: 0.3rem; font-size: 0.78rem; font-weight: 600; color: #3a6470; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }
.ripples::before, .ripples::after {
    content: ""; position: absolute; inset: 14px; border-radius: 50%; border: 2px solid #ef4444; pointer-events: none;
    animation: ripple 2.4s ease-out 1.4s infinite;
}
.ripples::after { animation-delay: 2.6s; }

.tile-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.8rem; margin: 1.2rem 0 1rem 0; }
.tile { border-radius: 16px; padding: 0.8rem 1rem; background: rgba(255,255,255,0.78); animation: popIn 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
.tile:nth-child(1) { animation-delay: 0.5s; } .tile:nth-child(2) { animation-delay: 0.62s; } .tile:nth-child(3) { animation-delay: 0.74s; }
.tile .t-label { display: block; font-size: 0.76rem; font-weight: 600; color: #3a6470; }
.tile .t-value { display: block; font-family: 'Sora', sans-serif; font-weight: 800; font-size: clamp(1.15rem, 3vw, 1.5rem); margin-top: 0.1rem; }
.bar-label { display: flex; justify-content: space-between; font-weight: 700; font-size: 0.88rem; margin: 0.55rem 0 0.25rem 0; color: #0b3b45; }
.bar-track { width: 100%; height: 12px; border-radius: 999px; background: rgba(15, 42, 51, 0.07); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; width: var(--w); animation: grow 1.5s cubic-bezier(0.22, 1, 0.36, 1) 0.6s backwards; }

/* ---- malignant ---- */
.alert-card {
    background: linear-gradient(145deg, rgba(254,242,242,0.94), rgba(255,247,237,0.94));
    border: 2px solid rgba(245,158,11,0.9);
    animation: shakeOnce 0.6s ease 0.1s both, pulseGlow 2.2s ease-in-out 0.7s infinite;
}
.alert-card .status-chip { background: linear-gradient(120deg, #ef4444, #f59e0b); }
.alert-card .result-title { color: #b91c1c; }
.alert-card .result-sub { color: #7f1d1d; }
.alert-card .tile { border: 1px solid rgba(239,68,68,0.28); }
.alert-card .tile .t-value { color: #b91c1c; }
.alert-card .tile.warn { border-color: rgba(239,68,68,0.7); animation: popIn 0.6s cubic-bezier(0.34,1.56,0.64,1) 0.5s both, blink 1.8s ease-in-out 1.3s infinite; }
.guidance {
    margin-top: 1.5rem; padding: clamp(1rem, 3vw, 1.4rem); border-radius: 18px;
    background: rgba(255,255,255,0.86); border-left: 6px solid #ef4444;
    box-shadow: 0 10px 30px rgba(239,68,68,0.14);
    animation: alertEnter 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.9s backwards;
}
.guidance h4 { margin: 0 0 0.6rem 0; color: #b91c1c; font-size: 1.1rem; }
.guidance ol { list-style: none; counter-reset: step; margin: 0; padding: 0; }
.guidance li {
    counter-increment: step; position: relative; padding: 0.55rem 0 0.55rem 2.7rem; color: #3b2a2a; line-height: 1.5;
    animation: alertEnter 0.7s cubic-bezier(0.22, 1, 0.36, 1) backwards;
}
.guidance li:nth-child(1) { animation-delay: 1.2s; } .guidance li:nth-child(2) { animation-delay: 1.4s; }
.guidance li:nth-child(3) { animation-delay: 1.6s; } .guidance li:nth-child(4) { animation-delay: 1.8s; }
.guidance li:nth-child(5) { animation-delay: 2.0s; }
.guidance li::before {
    content: counter(step); position: absolute; left: 0; top: 0.5rem; width: 1.9rem; height: 1.9rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; color: #fff;
    background: linear-gradient(135deg, #ef4444, #f59e0b);
}
.guidance li b { color: #7f1d1d; }

/* ---- benign ---- */
.safe-card {
    background: linear-gradient(145deg, rgba(236,253,245,0.96) 0%, rgba(209,250,229,0.92) 55%, rgba(204,251,241,0.92) 100%);
    border: 1.5px solid rgba(16,185,129,0.5); box-shadow: 0 18px 50px rgba(16,185,129,0.18);
    animation: fadeUp 0.7s ease both;
}
.safe-card::after {
    content: ""; position: absolute; top: 0; bottom: 0; left: 0; width: 22%; z-index: 0; pointer-events: none;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.75), transparent);
    animation: sweep 5.5s ease-in-out 1.2s infinite;
}
.sparks { position: absolute; inset: 0; z-index: 0; pointer-events: none; overflow: hidden; }
.sp {
    position: absolute; bottom: 0; left: var(--x); width: var(--s); height: var(--s); border-radius: 50%;
    background: radial-gradient(circle, #fff 0%, #34d399 55%, transparent 70%);
    animation: riseStar 4.5s ease-in var(--d) infinite both;
}
.safe-card .status-chip { background: linear-gradient(120deg, #10b981, #34d399); }
.safe-card .result-title { color: #047857; }
.safe-card .result-sub { color: #065f46; }
.safe-card .tile { border: 1px solid rgba(16,185,129,0.3); }
.safe-card .tile .t-value { color: #047857; }
.check { width: 46px; height: 46px; flex: 0 0 auto; }
.check circle { fill: #10b981; stroke: #059669; stroke-width: 2; animation: popIn 0.6s cubic-bezier(0.34,1.56,0.64,1) 0.2s both; transform-origin: center; }
.check path { fill: none; stroke: #fff; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 40; stroke-dashoffset: 40; animation: drawStroke 0.6s ease 0.9s forwards; }
.low-risk-badge {
    display: inline-flex; align-items: center; gap: 0.5rem; margin-top: 0.9rem;
    padding: 0.5rem 1.2rem; border-radius: 999px; font-weight: 800; font-size: 0.95rem;
    color: #fff; background: linear-gradient(120deg, #059669, #10b981);
    animation: popIn 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.8s both, badgeGlow 2.4s ease-out 1.6s infinite;
}
.safe-note { margin: 1.2rem 0 0 0; color: #065f46; line-height: 1.6; }

/* ---- uncertain ---- */
.unsure-card {
    background: linear-gradient(145deg, rgba(255,251,235,0.96), rgba(254,243,199,0.88));
    border: 1.5px solid rgba(217,119,6,0.5); box-shadow: 0 16px 40px rgba(245,158,11,0.18);
    animation: fadeUp 0.6s ease both;
}
.unsure-card .status-chip { background: linear-gradient(120deg, #f59e0b, #d97706); }
.unsure-card .result-title { color: #92400e; }
.unsure-card .result-sub { color: #92400e; }
.unsure-card .tile { border: 1px solid rgba(217,119,6,0.3); }
.unsure-card .tile .t-value { color: #92400e; }

/* ============ RESPONSIVE ============ */
.responsive-img img, [data-testid="stImage"] img { max-height: 440px; object-fit: contain; width: 100%; }
@media (max-width: 900px) {
    .block-container { padding-top: 5.5rem; }
    .topnav .links { display: none; }
    .mobile-menu { display: block; }
    .glass, .st-key-upload_card, .st-key-preview_card, .st-key-settings_card, .st-key-analyze_card { padding: 1.1rem; border-radius: 16px; }
}
@media (max-width: 700px) {
    .result-main { grid-template-columns: 1fr; justify-items: center; text-align: center; }
    .status-row { justify-content: center; }
    .tile-row { grid-template-columns: repeat(2, 1fr); }
    .tile-row .tile:last-child:nth-child(odd) { grid-column: 1 / -1; }
    [data-testid="stFileUploaderDropzone"] { flex-direction: column; text-align: center; }
    .guidance li { padding-left: 2.5rem; }
}
@media (max-width: 480px) {
    .block-container { padding-top: 5rem; padding-left: 0.8rem; padding-right: 0.8rem; }
    .hero { padding: 1.6rem 0.8rem; }
    .stButton > button { padding: 0.9rem 1rem; font-size: 1rem; }
    .topnav .brand { font-size: 0.95rem; }
}
/* touch screens: no hover lift (it sticks after a tap) */
@media (hover: none) {
    .glass:hover, .st-key-upload_card:hover, .st-key-preview_card:hover,
    .st-key-settings_card:hover, .st-key-analyze_card:hover { transform: none; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-delay: 0s !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}
</style>

<div class="stars"></div>
<div class="stars-2"></div>
<div class="shooting-star"></div>
<div class="nebula neb-1"></div><div class="nebula neb-2"></div><div class="nebula neb-3"></div>

<nav class="topnav">
    <a class="brand" href="#home"><span class="pulse-dot"></span>Skin Lesion Analyzer</a>
    <div class="links">
        <a class="nav-link" href="#home">Home</a>
        <a class="nav-link" href="#analyzer">Analyzer</a>
        <a class="nav-link" href="#settings">Settings</a>
        <a class="nav-link" href="#results">Results</a>
        <a class="nav-link" href="#about">About</a>
    </div>
    <details class="mobile-menu">
        <summary aria-label="Open menu">&#9776;</summary>
        <div class="drop">
            <a class="nav-link" href="#home">Home</a>
            <a class="nav-link" href="#analyzer">Analyzer</a>
            <a class="nav-link" href="#settings">Settings</a>
            <a class="nav-link" href="#results">Results</a>
            <a class="nav-link" href="#about">About</a>
        </div>
    </details>
</nav>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. RESULT CARD RENDERERS
# ----------------------------------------------------------------------------

def tiles_html(items):
    out = ""
    for label, value, extra in items:
        out += f'<div class="tile {extra}"><span class="t-label">{label}</span><span class="t-value">{value}</span></div>'
    return f'<div class="tile-row rise" style="--i:3">{out}</div>'


def bars_html(benign_p, malignant_p, b_color, m_color):
    return f"""
    <div class="rise" style="--i:4">
        <div class="bar-label"><span>Benign</span><span>{benign_p * 100:.1f}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="--w:{benign_p * 100:.1f}%; background:{b_color};"></div></div>
        <div class="bar-label"><span>Malignant</span><span>{malignant_p * 100:.1f}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="--w:{malignant_p * 100:.1f}%; background:{m_color};"></div></div>
    </div>
    """


def gauge_html(value, g1, g2, caption, ripples=False):
    pct = value * 100
    wrap = "gauge-wrap ripples" if ripples else "gauge-wrap"
    return f"""
    <div class="{wrap}">
        <div class="gauge" style="--target:{pct:.1f}; --tint:{int(round(pct))}; --g1:{g1}; --g2:{g2};">
            <div class="gauge-center">
                <span class="sr">{pct:.1f}% {caption}</span>
                <span class="num" aria-hidden="true"></span>
                <span class="cap">{caption}</span>
            </div>
        </div>
    </div>
    """


def render_malignant(benign_p, malignant_p, confidence, threshold):
    tiles = tiles_html([
        ("Malignant probability", f"{malignant_p * 100:.1f}%", "warn"),
        ("Confidence", f"{confidence * 100:.1f}%", ""),
        ("Threshold", f"{threshold:.2f}", ""),
    ])
    st.markdown(
        f"""
        <div class="result-card alert-card">
            <div class="result-main">
                {gauge_html(malignant_p, "#f59e0b", "#dc2626", "malignant", ripples=True)}
                <div class="result-info">
                    <div class="status-row rise" style="--i:0"><span class="status-chip">Needs attention</span></div>
                    <p class="result-title rise" style="--i:1">Cancer-like features detected</p>
                    <p class="result-sub rise" style="--i:2">The model classified this lesion as <b>malignant</b>. This is a screening result, not a diagnosis.</p>
                    {tiles}
                    {bars_html(benign_p, malignant_p, "linear-gradient(90deg,#94a3b8,#cbd5e1)", "linear-gradient(90deg,#f59e0b,#ef4444)")}
                </div>
            </div>
            <div class="guidance">
                <h4>What to do next</h4>
                <ol>
                    <li><b>Stay calm.</b> An AI result can be wrong. It is a reason to get checked, not a confirmed diagnosis.</li>
                    <li><b>Book a dermatologist</b> as soon as you can, ideally within the next few days.</li>
                    <li><b>Photograph the lesion</b> in good light with a ruler or coin for scale, and note any change in size, colour or shape.</li>
                    <li><b>Protect the area.</b> Avoid sun exposure, scratching, picking or home treatments.</li>
                    <li><b>Seek care urgently</b> if the lesion bleeds, itches, grows quickly or becomes painful.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_benign(benign_p, malignant_p, confidence, threshold):
    tiles = tiles_html([
        ("Benign probability", f"{benign_p * 100:.1f}%", ""),
        ("Confidence", f"{confidence * 100:.1f}%", ""),
        ("Threshold", f"{threshold:.2f}", ""),
    ])
    sparks = "".join(
        f'<i class="sp" style="--x:{x}%; --d:{d}s; --s:{sz}px"></i>'
        for x, d, sz in zip(
            range(6, 100, 10),
            [0, 0.8, 1.6, 0.4, 2.2, 1.2, 2.8, 0.2, 1.9, 0.6],
            [6, 9, 7, 10, 6, 8, 7, 9, 6, 8],
        )
    )
    st.markdown(
        f"""
        <div class="result-card safe-card">
            <div class="sparks">{sparks}</div>
            <div class="result-main">
                {gauge_html(benign_p, "#6ee7b7", "#059669", "benign")}
                <div class="result-info">
                    <div class="status-row rise" style="--i:0">
                        <svg class="check" viewBox="0 0 52 52" aria-hidden="true"><circle cx="26" cy="26" r="24"/><path d="M14 27l8 8 16-17"/></svg>
                        <span class="status-chip">No cancer detected</span>
                    </div>
                    <p class="result-title rise" style="--i:1">This lesion looks benign</p>
                    <p class="result-sub rise" style="--i:2">The model found no features that point to cancer.</p>
                    <span class="low-risk-badge">LOW RISK &middot; {benign_p * 100:.1f}% benign</span>
                    {tiles}
                    {bars_html(benign_p, malignant_p, "linear-gradient(90deg,#34d399,#10b981)", "linear-gradient(90deg,#cbd5e1,#94a3b8)")}
                </div>
            </div>
            <p class="safe-note rise" style="--i:5">Keep an eye on the spot. See a doctor if it changes in size, colour or shape, starts to bleed, or looks different from your other moles. Routine skin checks are still a good habit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uncertain(benign_p, malignant_p, confidence, threshold):
    tiles = tiles_html([
        ("Benign", f"{benign_p * 100:.1f}%", ""),
        ("Malignant", f"{malignant_p * 100:.1f}%", ""),
        ("Threshold", f"{threshold:.2f}", ""),
    ])
    st.markdown(
        f"""
        <div class="result-card unsure-card">
            <div class="result-main">
                {gauge_html(malignant_p, "#fbbf24", "#d97706", "malignant")}
                <div class="result-info">
                    <div class="status-row rise" style="--i:0"><span class="status-chip">Inconclusive</span></div>
                    <p class="result-title rise" style="--i:1">Result is uncertain</p>
                    <p class="result-sub rise" style="--i:2">Neither class passed your threshold. Please have this lesion looked at by a dermatologist.</p>
                    {tiles}
                    {bars_html(benign_p, malignant_p, "linear-gradient(90deg,#34d399,#10b981)", "linear-gradient(90deg,#f59e0b,#ef4444)")}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 4. HERO HEADER  (anchor: #home)
# ----------------------------------------------------------------------------

st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="glass hero intro-1">
        <h1 class="grad-text float-anim">Skin Lesion Analyzer</h1>
        <p class="sub">AI-assisted dermoscopic image classification — Benign vs. Malignant</p>
        <span class="disclaimer-pill">For research and educational use only — not a medical diagnosis</span>
        <div class="chips"><span class="chip">224 x 224 input</span><span class="chip">Processed in memory only</span><span class="chip">Adjustable threshold</span></div>
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
    with st.container(key="upload_card"):
        st.markdown("### Upload Lesion Image")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse — accepted formats: .jpg, .jpeg, .png",
            type=["jpg", "jpeg", "png"], label_visibility="collapsed",
        )

    analyze_clicked = False
    if uploaded_file is not None:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        with st.container(key="analyze_card"):
            analyze_clicked = st.button("Analyze Lesion", use_container_width=True)

with col_preview:
    with st.container(key="preview_card"):
        st.markdown("### Image Preview")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded lesion image", use_container_width=True)
            st.caption(f"Original {image.size[0]} x {image.size[1]} px — resized to 224 x 224 for inference")
        else:
            st.info("Upload an image to see a live preview here.")

# Reset stored result when a different file is uploaded
current_key = f"{uploaded_file.name}-{uploaded_file.size}" if uploaded_file is not None else None
if st.session_state.get("file_key") != current_key:
    st.session_state["file_key"] = current_key
    st.session_state["probs"] = None

st.markdown('<hr class="soft">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. SETTINGS  (anchor: #settings)
# ----------------------------------------------------------------------------

st.markdown('<div id="settings"></div>', unsafe_allow_html=True)
with st.container(key="settings_card"):
    st.markdown("#### Decision Threshold")
    c_set1, c_set2 = st.columns([3, 1])
    with c_set1:
        threshold = st.slider(
            "Malignant probability above this value is classified as Malignant.",
            min_value=0.05, max_value=0.95, value=0.50, step=0.01,
            label_visibility="collapsed",
        )
    with c_set2:
        st.metric("Threshold", f"{threshold:.2f}")

# ----------------------------------------------------------------------------
# 7. RESULTS  (anchor: #results)
# ----------------------------------------------------------------------------

st.markdown('<div id="results"></div>', unsafe_allow_html=True)
st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

if uploaded_file is not None and analyze_clicked:
    with st.spinner("Running deep-learning inference..."):
        time.sleep(0.4)
        img_batch = preprocess_image(image)
        label, confidence, probs = predict(model, img_batch)
        st.session_state["probs"] = [float(probs[0]), float(probs[1])]

if uploaded_file is not None and st.session_state.get("probs") is not None:
    benign_p, malignant_p = st.session_state["probs"]
    confidence = max(benign_p, malignant_p)

    if malignant_p >= threshold:
        render_malignant(benign_p, malignant_p, confidence, threshold)
    elif benign_p >= threshold:
        render_benign(benign_p, malignant_p, confidence, threshold)
    else:
        render_uncertain(benign_p, malignant_p, confidence, threshold)

elif uploaded_file is not None:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; color:#3a6470; margin:0;">'
        "Press <b>Analyze Lesion</b> to run the classifier.</p></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="glass fade-up"><p style="text-align:center; color:#3a6470; margin:0;">'
        "Upload a dermoscopic image to begin analysis.</p></div>",
        unsafe_allow_html=True,
    )

if analyze_clicked and st.session_state.get("probs") is not None:
    # smoothly bring the fresh result into view (important on phones)
    components.html(
        """<script>try{setTimeout(function(){var el=window.parent.document.getElementById('results');
        if(el){el.scrollIntoView({behavior:'smooth',block:'start'});}},400);}catch(e){}</script>""",
        height=0,
    )

# ----------------------------------------------------------------------------
# 8. ABOUT + GUIDELINES + FOOTER  (anchor: #about)
# ----------------------------------------------------------------------------

st.markdown('<div style="height:1.6rem"></div><div id="about"></div>', unsafe_allow_html=True)
c_a, c_b = st.columns(2, gap="large")

with c_a:
    st.markdown(
        """
        <div class="glass reveal" style="height:100%;">
            <h3 style="margin-top:0;">Image Guidelines</h3>
            <ul class="checklist">
                <li>Use a <b>clear, in-focus</b> close-up of the lesion</li>
                <li>Even, bright lighting — avoid harsh shadows</li>
                <li>Fill most of the frame with the lesion</li>
                <li>Accepted formats: <b>.jpg</b>, <b>.jpeg</b>, or <b>.png</b></li>
                <li>Avoid blurry, filtered, or heavily compressed images</li>
                <li>Hair, rulers, or ink marks may reduce accuracy</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_b:
    st.markdown(
        """
        <div class="glass reveal" style="height:100%;">
            <h3 style="margin-top:0;">About and Privacy</h3>
            <ul class="checklist">
                <li>Deep-learning classifier with a 224 x 224 input, built on TensorFlow</li>
                <li>Model weights auto-download once, then inference runs instantly</li>
                <li>Adjustable decision threshold for sensitivity control</li>
                <li>Images are processed in memory only — never stored</li>
                <li>Fully responsive — works on phone, tablet, and desktop</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="glass reveal" style="margin-top:1.6rem; text-align:center;">
        <small style="color:#3a6470;">
        <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified dermatologist for any skin concerns.
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
