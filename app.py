import os
import io
import time
import base64

import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =============================================================================
# 1. CONFIGURATION & MODEL
# =============================================================================

CLASS_LABELS = ["benign", "malignant"]
IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.keras"
GDRIVE_FILE_ID = "16tlVBG4-pbqmnWCUEaHCxtNCmQ_Y5lnS"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

st.set_page_config(
    page_title="DermaScan - Skin Lesion Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights, please wait..."):
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


def load_image(uploaded) -> Image.Image:
    img = Image.open(uploaded)
    return ImageOps.exif_transpose(img)


def img_to_b64(img: Image.Image, max_side: int = 760) -> str:
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def H(s: str) -> str:
    """Strip indentation so Streamlit's markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def html(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


# =============================================================================
# 2. ANIMATED SKIN CROSS-SECTION (SVG)
# =============================================================================

def skin_svg(p: str = "s") -> str:
    """Skin layers with a scanning beam, a pulsing lesion, swaying hair and flowing blood."""
    return f"""
    <svg class="skin" viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Animated cross-section of skin being scanned">
      <defs>
        <linearGradient id="{p}epi" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#F7D8C0"/><stop offset="1" stop-color="#E6B391"/></linearGradient>
        <linearGradient id="{p}der" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#DB8F7F"/><stop offset="1" stop-color="#B8606A"/></linearGradient>
        <linearGradient id="{p}hyp" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#EBBB76"/><stop offset="1" stop-color="#C98B4A"/></linearGradient>
        <linearGradient id="{p}beam" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#0EA5A4" stop-opacity="0"/>
          <stop offset="0.85" stop-color="#0EA5A4" stop-opacity="0.45"/>
          <stop offset="1" stop-color="#2DD4BF" stop-opacity="0.95"/></linearGradient>
        <radialGradient id="{p}les" cx="0.5" cy="0.35" r="0.7">
          <stop offset="0" stop-color="#7A4B3A"/><stop offset="1" stop-color="#2E1A15"/></radialGradient>
        <clipPath id="{p}clip"><rect width="420" height="320" rx="26"/></clipPath>
      </defs>
      <g clip-path="url(#{p}clip)">
        <rect width="420" height="320" fill="#EEF0FF"/>
        <path d="M0 205 Q70 190 140 205 T280 205 T420 200 V320 H0Z" fill="url(#{p}hyp)"/>
        <path d="M0 118 Q70 106 140 118 T280 118 T420 114 V212 Q350 228 280 212 T140 212 T0 218Z" fill="url(#{p}der)"/>
        <path d="M0 86 Q50 80 100 86 T200 86 T300 86 T420 86 V128 Q350 140 280 128 T140 128 T0 132Z" fill="url(#{p}epi)"/>
        <path d="M0 86 Q50 80 100 86 T200 86 T300 86 T420 86" fill="none" stroke="#FBE6D4" stroke-width="3"/>
        <path d="M-10 160 C50 146 110 176 180 160 S320 146 430 164" fill="none" stroke="#7F1D1D" stroke-opacity=".35" stroke-width="9"/>
        <path class="flow" d="M-10 160 C50 146 110 176 180 160 S320 146 430 164" fill="none" stroke="#FECACA" stroke-width="4" stroke-linecap="round"/>
        <path class="fiber" d="M10 190 q40 -14 80 0 t80 0 t80 0 t80 0" fill="none" stroke="#fff" stroke-opacity=".28" stroke-width="1.6"/>
        <path class="fiber" d="M0 140 q40 12 80 0 t80 0 t80 0 t80 0 t80 0" fill="none" stroke="#fff" stroke-opacity=".22" stroke-width="1.6"/>
        <path d="M312 150 L312 90" stroke="#4A2A2A" stroke-opacity=".55" stroke-width="7" stroke-linecap="round"/>
        <circle cx="312" cy="152" r="9" fill="#8A4040"/>
        <path d="M88 150 L88 90" stroke="#4A2A2A" stroke-opacity=".55" stroke-width="7" stroke-linecap="round"/>
        <circle cx="88" cy="152" r="9" fill="#8A4040"/>
        <path class="sway" d="M312 88 C310 62 322 44 336 30" fill="none" stroke="#3B2626" stroke-width="3.4" stroke-linecap="round"/>
        <path class="sway d2" d="M88 88 C86 64 76 46 64 34" fill="none" stroke="#3B2626" stroke-width="3.4" stroke-linecap="round"/>
        <circle class="cell" cx="46" cy="108" r="5" fill="#fff" fill-opacity=".38"/>
        <circle class="cell c2" cx="130" cy="112" r="4" fill="#fff" fill-opacity=".32"/>
        <circle class="cell c3" cx="268" cy="110" r="5" fill="#fff" fill-opacity=".36"/>
        <circle class="cell c4" cx="372" cy="108" r="4" fill="#fff" fill-opacity=".30"/>
        <ellipse cx="210" cy="104" rx="30" ry="22" fill="#5B3A2E" fill-opacity=".55"/>
        <path d="M172 88 Q210 38 248 88Z" fill="url(#{p}les)"/>
        <ellipse class="ping" cx="210" cy="78" rx="46" ry="26" fill="none" stroke="#0EA5A4" stroke-width="2.5"/>
        <ellipse class="ping p2" cx="210" cy="78" rx="46" ry="26" fill="none" stroke="#0EA5A4" stroke-width="2"/>
        <g class="beam">
          <rect x="-70" y="20" width="70" height="300" fill="url(#{p}beam)"/>
          <rect x="-2" y="20" width="2.5" height="300" fill="#CCFBF1"/>
        </g>
        <text x="14" y="112" font-size="11" font-weight="700" fill="#4A2616" fill-opacity=".9" font-family="Inter,sans-serif">Epidermis</text>
        <text x="14" y="186" font-size="11" font-weight="700" fill="#fff" fill-opacity=".92" font-family="Inter,sans-serif">Dermis</text>
        <text x="14" y="262" font-size="11" font-weight="700" fill="#4A2D0E" fill-opacity=".9" font-family="Inter,sans-serif">Hypodermis</text>
        <text x="210" y="30" text-anchor="middle" font-size="11" font-weight="700" fill="#0F766E" font-family="Inter,sans-serif">Lesion detected</text>
      </g>
    </svg>
    """


# =============================================================================
# 3. CLINICAL GUIDANCE CONTENT
# =============================================================================

GUIDANCE = {
    "malignant": {
        "title": "Malignant features detected",
        "tone": "danger",
        "summary": (
            "The model found patterns that are commonly seen in skin cancer. "
            "This is not a diagnosis, but this lesion should be examined by a doctor soon."
        ),
        "level": 4,
        "level_name": "High concern",
        "danger": (
            "Malignant skin lesions, especially melanoma, can grow deeper into the skin and spread "
            "to lymph nodes and other organs if left untreated. The good news: when found early, "
            "most skin cancers are highly treatable. Time matters more than anything else here."
        ),
        "doctor_head": "Yes - see a dermatologist as soon as possible",
        "doctor_when": "Aim for an appointment within 1-2 weeks.",
        "doctor_points": [
            "Go the same week if the spot bleeds, itches, crusts, or is growing quickly.",
            "Bring this result and, if you can, earlier photos of the spot.",
            "Ask for a full-body skin check, not only this one spot.",
        ],
        "treatment_head": "What treatment may involve",
        "treatment": [
            ("Biopsy", "A small sample is tested in a lab to confirm what the lesion is."),
            ("Surgical removal", "The most common treatment for early-stage skin cancer."),
            ("Mohs surgery", "Layer-by-layer removal, used for some types and sensitive areas such as the face."),
            ("Advanced treatment", "If the cancer has spread: immunotherapy, targeted therapy, or radiation, planned by a cancer team."),
        ],
        "next": [
            "Book a dermatologist appointment today.",
            "Take a clear photo of the spot with a coin or ruler beside it for size.",
            "Avoid picking, scratching, or applying creams to it.",
            "Protect the area from sun until you are seen.",
        ],
    },
    "uncertain": {
        "title": "Result is uncertain",
        "tone": "warn",
        "summary": (
            "The model could not clearly decide between benign and malignant. "
            "An unclear result is a reason to get a professional opinion, not a reason to panic."
        ),
        "level": 3,
        "level_name": "Moderate - needs a check",
        "danger": (
            "Uncertain results can come from low image quality, unusual lesion shapes, or features "
            "that overlap between harmless and harmful spots. The risk cannot be judged from this "
            "image alone."
        ),
        "doctor_head": "Recommended - book a skin check",
        "doctor_when": "Aim for an appointment within 2-4 weeks, sooner if the spot changes.",
        "doctor_points": [
            "A dermatologist can examine it with a dermatoscope in minutes.",
            "Retake the photo in bright, even light and upload it again for a second opinion.",
            "Go sooner if it grows, changes colour, bleeds, or starts to itch.",
        ],
        "treatment_head": "What a doctor may do",
        "treatment": [
            ("Dermoscopy exam", "Magnified, in-person inspection of the lesion."),
            ("Photo monitoring", "Regular photos to track any change over weeks or months."),
            ("Biopsy", "Only if the doctor finds the spot suspicious."),
        ],
        "next": [
            "Retake the photo with better focus and lighting.",
            "Book a skin check with a dermatologist or GP.",
            "Photograph the spot weekly to track changes.",
            "Use sunscreen (SPF 30+) and avoid sunburn.",
        ],
    },
    "benign": {
        "title": "Looks benign",
        "tone": "good",
        "summary": (
            "The model found patterns that are more typical of a harmless lesion. "
            "A benign result from an app does not completely rule out skin cancer, so keep watching the spot."
        ),
        "level": 1,
        "level_name": "Low concern",
        "danger": (
            "Benign lesions such as common moles do not spread and are generally harmless. "
            "A small number of cancers can look benign, so changes over time are what matter most."
        ),
        "doctor_head": "Not urgent - routine check is enough",
        "doctor_when": "Have your skin checked at your regular yearly visit.",
        "doctor_points": [
            "See a doctor sooner if the spot changes in size, shape, or colour.",
            "See a doctor if it starts bleeding, itching, or becoming painful.",
            "People with fair skin, many moles, or a family history should check more often.",
        ],
        "treatment_head": "Treatment",
        "treatment": [
            ("Usually none needed", "Harmless lesions typically do not need treatment."),
            ("Optional removal", "Can be removed for comfort, rubbing, or cosmetic reasons."),
            ("Sun protection", "SPF 30+, hats, and shade lower the risk of new lesions."),
        ],
        "next": [
            "Photograph the spot now so you can compare later.",
            "Re-check it once a month using the ABCDE signs below.",
            "Wear sunscreen and avoid tanning beds.",
            "Keep your yearly skin check.",
        ],
    },
}

ABCDE = [
    ("A", "Asymmetry", "One half does not match the other."),
    ("B", "Border", "Edges are ragged, blurred, or notched."),
    ("C", "Colour", "Several shades of brown, black, red, white, or blue."),
    ("D", "Diameter", "Wider than about 6 mm (a pencil eraser)."),
    ("E", "Evolving", "Any change in size, shape, colour, or feel."),
]

# =============================================================================
# 4. THEME  -  light: misty lavender background, navy text, indigo buttons
# =============================================================================
# Palette
#   page        #F6F7FF -> #E7F3F6   soft mist
#   ink         #16213E               deep navy text
#   muted       #4A5578               secondary text
#   primary     #4338CA               indigo buttons, white text
#   aqua        #0EA5A4               scan beam / highlights
#   good #15803D   warn #B45309   danger #B91C1C

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
  color-scheme:light;
  --ink:#16213E; --muted:#4A5578; --primary:#4338CA; --primary-hi:#4338CA; --on-primary:#FFFFFF;
  --aqua:#0EA5A4; --line:rgba(67,56,202,.16); --card:rgba(255,255,255,.80); --soft:#F5F6FF;
  --good:#15803D; --good-bg:#E8F7EE; --good-ln:#86D5A3;
  --warn:#B45309; --warn-bg:#FEF3E2; --warn-ln:#F5C98A;
  --danger:#B91C1C; --danger-bg:#FDECEC; --danger-ln:#F4A3A3;
}

html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family:'Inter',sans-serif; color:var(--ink);
  background:
    radial-gradient(ellipse 80% 55% at 10% -5%, rgba(129,140,248,.28), transparent 60%),
    radial-gradient(ellipse 70% 50% at 95% 10%, rgba(45,212,191,.22), transparent 55%),
    linear-gradient(170deg,#F6F7FF 0%,#EDF4FB 55%,#E7F3F6 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], footer, #MainMenu{display:none !important;}
.block-container{max-width:1180px; padding:1rem 1.5rem 3rem 1.5rem;}
h1,h2,h3,h4{font-family:'Sora',sans-serif; color:var(--ink); letter-spacing:-.3px;}
p,li,label,small,[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"]{color:var(--muted);}
b,strong,[data-testid="stMarkdownContainer"] strong{color:var(--ink);}

/* ---------- ambient background (decorative) ---------- */
.orb{position:fixed; border-radius:50%; filter:blur(100px); pointer-events:none; z-index:0;
     animation:drift 26s ease-in-out infinite alternate;}
.orb.o1{width:420px;height:420px;top:-130px;left:-130px;background:#A5B4FC;opacity:.45;}
.orb.o2{width:380px;height:380px;bottom:-130px;right:-100px;background:#99F6E4;opacity:.45;animation-delay:-9s;}
.orb.o3{width:260px;height:260px;top:50%;left:55%;background:#FBCFE8;opacity:.30;animation-delay:-16s;}
@keyframes drift{0%{transform:translate(0,0) scale(1);}50%{transform:translate(60px,40px) scale(1.1);}100%{transform:translate(-40px,70px) scale(.95);}}
.dust{position:fixed; top:0; left:0; width:3px; height:3px; border-radius:50%; z-index:0; pointer-events:none;
  background:transparent; animation:twinkle 4.5s ease-in-out infinite;
  box-shadow: 8vw 12vh #818CF8, 22vw 70vh #2DD4BF, 37vw 22vh #818CF8, 52vw 85vh #2DD4BF,
              66vw 14vh #818CF8, 79vw 58vh #2DD4BF, 91vw 33vh #818CF8, 14vw 44vh #2DD4BF,
              45vw 52vh #818CF8, 84vw 90vh #2DD4BF, 60vw 40vh #818CF8, 30vw 92vh #2DD4BF;}
@keyframes twinkle{0%,100%{opacity:.15;transform:scale(.9);}50%{opacity:.8;transform:scale(1.2);}}

/* ---------- cards (Streamlit containers with key="card_*") ---------- */
[class*="st-key-card"]{
  position:relative; z-index:1; background:var(--card);
  backdrop-filter:blur(14px) saturate(1.2); -webkit-backdrop-filter:blur(14px) saturate(1.2);
  border:1px solid var(--line); border-radius:22px; padding:1.5rem 1.6rem;
  box-shadow:0 10px 34px rgba(40,50,120,.12), inset 0 1px 0 #fff;
  animation:rise .6s cubic-bezier(.22,1,.36,1) both;
}
@keyframes rise{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:none;}}

/* ---------- NAVBAR: always one tidy row, on every screen size ---------- */
.st-key-topnav{
  position:relative; z-index:5; background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:.55rem .8rem; box-shadow:0 8px 26px rgba(40,50,120,.12);
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
}
.st-key-topnav [data-testid="stHorizontalBlock"]{
  flex-wrap:nowrap !important; align-items:center !important; gap:.5rem !important;
}
.st-key-topnav [data-testid="stColumn"], .st-key-topnav [data-testid="column"]{
  min-width:0 !important; width:auto !important; flex:0 0 auto !important;
}
.st-key-topnav [data-testid="stColumn"]:first-child, .st-key-topnav [data-testid="column"]:first-child{flex:1 1 auto !important;}
.st-key-topnav .stButton{width:auto !important;}
.st-key-topnav .stButton>button{width:auto !important; min-width:96px; padding:.55rem 1.1rem; font-size:.95rem; border-radius:12px; white-space:nowrap;}
.brand{display:flex;align-items:center;gap:.65rem;font-family:'Sora',sans-serif;font-weight:800;font-size:1.15rem;color:var(--ink);white-space:nowrap;}
.brand .logo{width:34px;height:34px;flex:0 0 34px;border-radius:11px;display:grid;place-items:center;position:relative;
  background:linear-gradient(135deg,#4338CA,#0EA5A4);box-shadow:0 6px 18px rgba(67,56,202,.35);}
.brand .logo::after{content:"";position:absolute;inset:-5px;border-radius:14px;border:2px solid rgba(67,56,202,.35);animation:ring 2.4s ease-out infinite;}
.brand .logo i{width:12px;height:12px;border-radius:50%;background:#fff;display:block;}
@keyframes ring{0%{transform:scale(.9);opacity:.9;}100%{transform:scale(1.4);opacity:0;}}

/* ---------- hero ---------- */
.hero{position:relative;z-index:1;display:grid;grid-template-columns:1.1fr 1fr;gap:2rem;align-items:center;padding:2rem .4rem 1rem;}
.hero h1{font-size:clamp(2rem,4.6vw,3.3rem);font-weight:800;margin:0 0 .8rem;line-height:1.1;}
.hero h1 .shine{background:linear-gradient(90deg,#4338CA,#0EA5A4,#4338CA);background-size:250% auto;
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 6s linear infinite;}
@keyframes shine{to{background-position:250% center;}}
.hero p{font-size:clamp(.98rem,2.2vw,1.12rem);line-height:1.65;margin:0;max-width:520px;}
.pill{display:inline-block;margin-top:1.1rem;padding:.45rem 1.1rem;border-radius:999px;font-size:.82rem;font-weight:600;
  color:#7C2D12;background:#FFEDD5;border:1px solid #FDBA74;}
.skinwrap{border-radius:26px;padding:6px;background:linear-gradient(135deg,rgba(67,56,202,.45),rgba(14,165,164,.40));box-shadow:0 16px 44px rgba(40,50,120,.22);}
.skinwrap svg{display:block;width:100%;height:auto;border-radius:22px;}

/* ---------- SKIN ANIMATION ---------- */
.skin .beam{animation:beam 4.6s ease-in-out infinite alternate;}
@keyframes beam{from{transform:translateX(0);}to{transform:translateX(490px);}}
.skin .sway{transform-box:fill-box;transform-origin:50% 100%;animation:sway 3.4s ease-in-out infinite alternate;}
.skin .sway.d2{animation-delay:-1.4s;}
@keyframes sway{from{transform:rotate(-6deg);}to{transform:rotate(7deg);}}
.skin .ping{transform-box:fill-box;transform-origin:center;animation:ping 2.6s ease-out infinite;}
.skin .ping.p2{animation-delay:1.3s;}
@keyframes ping{0%{transform:scale(.75);opacity:.95;}100%{transform:scale(1.5);opacity:0;}}
.skin .flow{stroke-dasharray:8 26;animation:flow 2.2s linear infinite;}
@keyframes flow{to{stroke-dashoffset:-68;}}
.skin .fiber{stroke-dasharray:6 8;animation:flow 6s linear infinite;}
.skin .cell{animation:bob 3.6s ease-in-out infinite alternate;}
.skin .cell.c2{animation-delay:-1s;}.skin .cell.c3{animation-delay:-2s;}.skin .cell.c4{animation-delay:-2.8s;}
@keyframes bob{from{transform:translateY(0);}to{transform:translateY(-7px);}}

/* ---------- stepper ---------- */
.steps{position:relative;z-index:1;display:flex;justify-content:center;align-items:center;gap:.6rem;margin:.6rem 0 1.4rem;}
.step{display:flex;align-items:center;gap:.5rem;padding:.45rem 1rem .45rem .5rem;border-radius:999px;background:var(--soft);
  border:1px solid var(--line);font-weight:600;font-size:.9rem;color:var(--muted);transition:all .4s ease;white-space:nowrap;}
.step b{width:26px;height:26px;flex:0 0 26px;border-radius:50%;display:grid;place-items:center;font-size:.8rem;background:#E0E4F8;color:var(--primary);}
.step.on{color:var(--ink);border-color:var(--primary);box-shadow:0 0 0 4px rgba(67,56,202,.12);}
.step.on b{background:var(--primary);color:var(--on-primary);}
.step.done b{background:var(--good);color:#fff;}
.step-line{width:26px;height:2px;background:var(--line);}

/* ---------- uploader ---------- */
[data-testid="stFileUploader"] section{
  background:#F5F6FF; border:2px dashed rgba(67,56,202,.45); border-radius:18px; padding:1.3rem; transition:all .3s ease;}
[data-testid="stFileUploader"] section:hover{border-color:var(--primary);background:#EEF0FF;box-shadow:0 0 0 5px rgba(67,56,202,.10);}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] div{color:var(--muted);}
[data-testid="stFileUploader"] svg{fill:var(--primary-hi);}
[data-testid="stFileUploader"] button{
  background:#fff !important;color:var(--primary) !important;border:2px solid var(--primary) !important;border-radius:12px !important;font-weight:700 !important;}
[data-testid="stFileUploader"] button:hover{background:var(--primary) !important;color:var(--on-primary) !important;}

/* ---------- BUTTONS (soft dark page): lavender fill + dark text; secondary = outlined light text ---------- */
.stButton>button{
  width:100%; border-radius:14px; font-weight:700; font-size:1.02rem; padding:.85rem 1.2rem;
  transition:transform .2s ease, box-shadow .2s ease, background .2s ease;
}
.stButton>button p{font-weight:700; margin:0;}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(120deg,#4338CA,#3730A3); color:#fff !important; border:2px solid #3730A3;
  box-shadow:0 8px 22px rgba(67,56,202,.35);
}
.stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p{color:var(--on-primary) !important;}
.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="stBaseButton-primary"]:hover{
  transform:translateY(-2px); box-shadow:0 12px 28px rgba(67,56,202,.45); background:linear-gradient(120deg,#4F46E5,#3730A3);
}
.stButton>button[kind="primary"]:active{transform:translateY(0) scale(.98);}
.stButton>button[kind="secondary"], .stButton>button[data-testid="stBaseButton-secondary"]{
  background:#fff; color:var(--primary) !important; border:2px solid var(--primary);
}
.stButton>button[kind="secondary"] p, .stButton>button[data-testid="stBaseButton-secondary"] p{color:var(--primary) !important;}
.stButton>button[kind="secondary"]:hover, .stButton>button[data-testid="stBaseButton-secondary"]:hover{
  background:#EEF0FF; transform:translateY(-1px);
}
.stButton>button:disabled{background:#D9DCEB !important;border-color:#C5C9DE !important;box-shadow:none !important;transform:none !important;}
.stButton>button:disabled p{color:#5B6385 !important;}
.stButton>button:focus-visible{outline:3px solid var(--aqua); outline-offset:3px;}

/* ---------- slider ---------- */
[data-baseweb="slider"] div[role="slider"]{background:var(--primary) !important;border-color:#fff !important;box-shadow:0 0 0 4px rgba(67,56,202,.25) !important;}
[data-testid="stSliderThumbValue"]{color:var(--primary-hi) !important;font-weight:700;}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{color:var(--muted) !important;}
.stSlider label{color:var(--ink) !important;}

/* ---------- image box ---------- */
.imgbox{position:relative;border-radius:18px;overflow:hidden;border:1px solid var(--line);box-shadow:0 10px 28px rgba(40,50,120,.18);background:#EEF0FF;}
.imgbox img{display:block;width:100%;height:auto;max-height:420px;object-fit:cover;}
.imgbox.scan::after{content:"";position:absolute;left:0;right:0;height:56px;top:-60px;
  background:linear-gradient(180deg,rgba(14,165,164,0),rgba(14,165,164,.45),rgba(14,165,164,0));
  border-bottom:2px solid #0EA5A4;animation:scan 3.2s ease-in-out infinite;}
@keyframes scan{0%{top:-60px;}100%{top:100%;}}
.corner{position:absolute;width:26px;height:26px;border:3px solid #fff;filter:drop-shadow(0 1px 3px rgba(0,0,0,.5));}
.corner.tl{top:12px;left:12px;border-right:0;border-bottom:0;border-radius:8px 0 0 0;}
.corner.tr{top:12px;right:12px;border-left:0;border-bottom:0;border-radius:0 8px 0 0;}
.corner.bl{bottom:12px;left:12px;border-right:0;border-top:0;border-radius:0 0 0 8px;}
.corner.br{bottom:12px;right:12px;border-left:0;border-top:0;border-radius:0 0 8px 0;}
.empty{display:grid;place-items:center;text-align:center;min-height:260px;border-radius:18px;border:2px dashed rgba(67,56,202,.25);background:#F7F8FF;padding:1rem;}
.empty .dot{width:54px;height:54px;border-radius:50%;margin:0 auto .8rem;background:radial-gradient(circle,#818CF8,#4338CA);animation:breathe 2.6s ease-in-out infinite;}
@keyframes breathe{0%,100%{transform:scale(.9);box-shadow:0 0 0 0 rgba(67,56,202,.4);}50%{transform:scale(1.05);box-shadow:0 0 0 18px rgba(67,56,202,0);}}

/* ---------- analyzing ---------- */
.analyzing{position:relative;z-index:1;text-align:center;padding:1.6rem 1rem;}
.analyzing .skinwrap{max-width:380px;margin:0 auto 1.2rem;}
.analyzing h3{margin:0 0 .3rem;}

/* ---------- result visuals ---------- */
.badge{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1.3rem;border-radius:999px;font-family:'Sora',sans-serif;font-weight:800;font-size:1.1rem;
  animation:pop .6s cubic-bezier(.34,1.56,.64,1) both;}
.badge::before{content:"";width:11px;height:11px;border-radius:50%;background:currentColor;animation:pulse 1.8s ease-out infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor;}70%{box-shadow:0 0 0 10px transparent;}100%{box-shadow:0 0 0 0 transparent;}}
@keyframes pop{0%{opacity:0;transform:scale(.8);}100%{opacity:1;transform:scale(1);}}
.badge.good{background:var(--good-bg);color:var(--good);border:2px solid var(--good-ln);}
.badge.warn{background:var(--warn-bg);color:var(--warn);border:2px solid var(--warn-ln);}
.badge.danger{background:var(--danger-bg);color:var(--danger);border:2px solid var(--danger-ln);}
.res-title{font-size:clamp(1.5rem,3.6vw,2.1rem);font-weight:800;margin:.9rem 0 .4rem;color:var(--ink);font-family:'Sora',sans-serif;}
.res-sum{font-size:1.04rem;line-height:1.65;max-width:560px;margin:0;}

.gauge{position:relative;width:170px;height:170px;margin:0 auto;}
.gauge svg{transform:rotate(-90deg);}
.gauge .trk{fill:none;stroke:#E3E6F6;stroke-width:12;}
.gauge .val{fill:none;stroke-width:12;stroke-linecap:round;stroke-dasharray:339.3;stroke-dashoffset:339.3;
  animation:fillring 1.6s cubic-bezier(.22,1,.36,1) .3s forwards;}
@keyframes fillring{to{stroke-dashoffset:var(--off);}}
.gauge .txt{position:absolute;inset:0;display:grid;place-content:center;text-align:center;animation:pop .8s ease .8s both;}
.gauge .txt b{font-family:'Sora',sans-serif;font-size:2.1rem;line-height:1;}
.gauge .txt small{font-size:.8rem;font-weight:600;}

.meter{display:flex;gap:6px;margin:.4rem 0 .8rem;}
.meter i{flex:1;height:14px;border-radius:7px;background:#E3E6F6;transform-origin:left;animation:grow .5s ease both;}
.meter i.on.good{background:#22A559;}.meter i.on.warn{background:#E8890C;}.meter i.on.danger{background:#DC2626;}
@keyframes grow{from{transform:scaleX(0);}to{transform:scaleX(1);}}

.bar{width:100%;height:16px;border-radius:999px;background:#E3E6F6;overflow:hidden;margin:.3rem 0 1rem;}
.bar>div{height:100%;border-radius:999px;width:0;animation:barfill 1.4s cubic-bezier(.22,1,.36,1) .3s forwards;}
@keyframes barfill{to{width:var(--w);}}
.row{display:flex;justify-content:space-between;font-weight:600;color:var(--ink);}
.row span{color:var(--ink);}

.card-h{display:flex;align-items:center;gap:.6rem;margin:0 0 .8rem;}
.card-h .ic{width:34px;height:34px;flex:0 0 34px;border-radius:10px;display:grid;place-items:center;font-size:1.05rem;background:#EEF0FF;}
.card-h h4{margin:0;font-size:1.08rem;}
.lead{font-weight:700;color:var(--ink) !important;font-size:1.05rem;margin:.2rem 0;}
.list{margin:.4rem 0 0;padding-left:1.15rem;}.list li{margin:.4rem 0;line-height:1.55;color:var(--muted);}
.tx{display:flex;gap:.7rem;align-items:flex-start;padding:.6rem .7rem;border-radius:12px;background:var(--soft);margin:.45rem 0;}
.tx .n{min-width:8px;height:8px;border-radius:50%;background:var(--primary);margin-top:.5rem;}
.tx p{margin:0;line-height:1.5;}.tx p b{display:block;}
.callout{padding:.9rem 1.1rem;border-radius:14px;margin-top:1rem;font-weight:500;line-height:1.55;color:var(--ink);}
.callout.good{background:var(--good-bg);border-left:5px solid var(--good);}
.callout.warn{background:var(--warn-bg);border-left:5px solid var(--warn);}
.callout.danger{background:var(--danger-bg);border-left:5px solid var(--danger);}
.abcde{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;}
.abcde div{background:var(--soft);border-radius:16px;padding:1rem .8rem;text-align:center;border:1px solid var(--line);transition:transform .25s ease, box-shadow .25s ease;}
.abcde div:hover{transform:translateY(-4px);box-shadow:0 10px 22px rgba(67,56,202,.15);}
.abcde .l{font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:800;color:var(--primary-hi);}
.abcde b{display:block;margin:.1rem 0 .3rem;}.abcde span{font-size:.86rem;line-height:1.4;display:block;}
.disclaimer{position:relative;z-index:1;text-align:center;font-size:.85rem;line-height:1.6;margin-top:1.4rem;padding:1rem 1.2rem;border-radius:16px;
  background:rgba(255,255,255,.7);border:1px solid var(--line);}

/* ---------- TABLET / PHONE ---------- */
@media (max-width:900px){
  .block-container{padding:.8rem .9rem 2.5rem;}
  [class*="st-key-card"]{padding:1.1rem;border-radius:18px;}
  .hero{grid-template-columns:1fr;text-align:center;gap:1.2rem;padding-top:1.4rem;}
  .hero p{margin:0 auto;}
  .abcde{grid-template-columns:repeat(2,1fr);}
  .abcde div:last-child{grid-column:1 / -1;}
  .res-center{text-align:center;}
  .res-sum{margin:0 auto;}
}
@media (max-width:560px){
  .brand{font-size:1rem;gap:.5rem;}
  .st-key-topnav{padding:.45rem .6rem;}
  .st-key-topnav .stButton>button{min-width:0;padding:.5rem .8rem;font-size:.86rem;}
  .steps{gap:.35rem;}
  .step-line{display:none;}
  .step{flex:1;justify-content:center;padding:.4rem .3rem;font-size:.74rem;gap:.3rem;}
  .step b{width:20px;height:20px;flex-basis:20px;font-size:.7rem;}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition:none !important;}}
</style>
<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div><div class="dust"></div>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# =============================================================================
# 5. SESSION STATE & NAVIGATION
# =============================================================================

ss = st.session_state
ss.setdefault("page", "detect")
ss.setdefault("result", None)


def go(page: str):
    ss.page = page


def top_bar():
    with st.container(key="topnav"):
        c1, c2, c3 = st.columns([1, 1, 1], gap="small", vertical_alignment="center")
        with c1:
            html('<div class="brand"><span class="logo"><i></i></span>DermaScan</div>')
        with c2:
            st.button("Analyze", key="nav_detect", on_click=go, args=("detect",),
                      type="primary" if ss.page == "detect" else "secondary")
        with c3:
            st.button("Results", key="nav_results", on_click=go, args=("results",),
                      type="primary" if ss.page == "results" else "secondary",
                      disabled=ss.result is None)


def stepper(active: int):
    items = ["Upload", "Analyze", "Results"]
    out = []
    for i, label in enumerate(items, 1):
        cls = "done" if i < active else ("on" if i == active else "")
        mark = "&#10003;" if i < active else str(i)
        out.append(f'<div class="step {cls}"><b>{mark}</b>{label}</div>')
        if i < len(items):
            out.append('<div class="step-line"></div>')
    html(f'<div class="steps">{"".join(out)}</div>')


def disclaimer():
    html("""
    <div class="disclaimer"><b>Medical disclaimer:</b> DermaScan is for educational and research use only.
    It does not replace professional medical advice, diagnosis, or treatment.
    Always ask a qualified dermatologist about any skin concern.</div>
    """)


# =============================================================================
# 6. PAGE 1 - ANALYZE
# =============================================================================

def page_detect():
    top_bar()
    html(f"""
    <div class="hero">
        <div>
            <h1>Check a skin spot<br><span class="shine">in three simple steps</span></h1>
            <p>Upload a close-up photo of a skin lesion. Our AI model estimates whether it looks
            benign or malignant, then explains what to do next.</p>
            <span class="pill">Research and education tool - not a medical diagnosis</span>
        </div>
        <div class="skinwrap">{skin_svg("h")}</div>
    </div>
    """)

    model = load_model()
    uploaded = ss.get("uploader")
    stepper(2 if uploaded is not None else 1)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        with st.container(key="card_upload"):
            html('<div class="card-h"><span class="ic">&#128247;</span><h4>Upload your photo</h4></div>')
            uploaded = st.file_uploader(
                "Upload a lesion image (jpg, jpeg or png)", type=["jpg", "jpeg", "png"],
                key="uploader", label_visibility="collapsed",
            )
            html("""
            <ul class="list">
                <li>Sharp, in-focus close-up with the spot filling most of the frame</li>
                <li>Bright, even light - avoid shadows and flash glare</li>
                <li>No filters, and keep hair or rulers out of the spot if you can</li>
            </ul>
            """)

    with col_r:
        with st.container(key="card_preview"):
            html('<div class="card-h"><span class="ic">&#128269;</span><h4>Your image</h4></div>')
            if uploaded is not None:
                image = load_image(uploaded)
                b64 = img_to_b64(image)
                html(f"""
                <div class="imgbox scan">
                    <img src="data:image/jpeg;base64,{b64}" alt="Uploaded skin lesion">
                    <span class="corner tl"></span><span class="corner tr"></span>
                    <span class="corner bl"></span><span class="corner br"></span>
                </div>
                """)
                st.caption(f"{image.size[0]} x {image.size[1]} px - resized to 224 x 224 for analysis")
            else:
                html("""
                <div class="empty"><div>
                    <div class="dot"></div>
                    <p class="lead">No image yet</p>
                    <p>Upload a photo and it will appear here.</p>
                </div></div>
                """)

    st.write("")
    with st.container(key="card_settings"):
        html('<div class="card-h"><span class="ic">&#9881;</span><h4>Sensitivity (optional)</h4></div>')
        st.markdown("Lower values flag more lesions as malignant (safer, but more false alarms). "
                    "The default of 0.50 works well for most photos.")
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                              key="threshold", label_visibility="collapsed")
        st.markdown(f"Current threshold: **{threshold:.2f}**")

    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        clicked = st.button("Analyze lesion", key="analyze", type="primary",
                            disabled=uploaded is None)
        if uploaded is None:
            st.caption("Upload an image first to enable analysis.")

    if clicked and uploaded is not None:
        slot = st.empty()
        slot.markdown(H(f"""
        <div class="analyzing">
            <div class="skinwrap">{skin_svg("a")}</div>
            <h3>Analyzing your image...</h3>
            <p>Scanning texture, colour, and border patterns. This takes a few seconds.</p>
        </div>
        """), unsafe_allow_html=True)

        image = load_image(uploaded)
        batch = preprocess_image(image)
        _, confidence, probs = predict(model, batch)
        time.sleep(1.2)

        benign_p, malignant_p = float(probs[0]), float(probs[1])
        if malignant_p >= threshold:
            key = "malignant"
        elif benign_p >= threshold:
            key = "benign"
        else:
            key = "uncertain"

        ss.result = {
            "key": key, "benign_p": benign_p, "malignant_p": malignant_p,
            "confidence": confidence, "threshold": threshold,
            "img_b64": img_to_b64(image), "size": image.size,
            "name": getattr(uploaded, "name", "image"),
        }
        ss.page = "results"
        slot.empty()
        st.rerun()

    disclaimer()


# =============================================================================
# 7. PAGE 2 - RESULTS
# =============================================================================

def page_results():
    top_bar()
    r = ss.result
    if r is None:
        st.write("")
        with st.container(key="card_empty"):
            html('<div class="empty"><div><div class="dot"></div><p class="lead">No results yet</p>'
                 '<p>Go to Analyze, upload an image, and press Analyze lesion.</p></div></div>')
            st.button("Go to Analyze", key="empty_go", on_click=go, args=("detect",), type="primary")
        disclaimer()
        return

    g = GUIDANCE[r["key"]]
    tone = g["tone"]
    ring = {"good": "#22A559", "warn": "#E8890C", "danger": "#DC2626"}[tone]
    st.write("")
    stepper(4)

    # ---- headline result ----
    offset = 339.3 * (1 - r["confidence"])
    with st.container(key="card_result"):
        c_img, c_txt, c_gauge = st.columns([1.15, 1.6, 1], gap="large", vertical_alignment="center")
        with c_img:
            html(f"""
            <div class="imgbox" style="border:3px solid {ring};">
                <img src="data:image/jpeg;base64,{r['img_b64']}" alt="Analyzed lesion">
            </div>
            """)
        with c_txt:
            badge_text = {"malignant": "Malignant", "benign": "Benign", "uncertain": "Uncertain"}[r["key"]]
            html(f"""
            <div class="res-center">
            <span class="badge {tone}">{badge_text}</span>
            <div class="res-title">{g['title']}</div>
            <p class="res-sum">{g['summary']}</p>
            </div>
            """)
        with c_gauge:
            html(f"""
            <div class="gauge">
                <svg width="170" height="170" viewBox="0 0 120 120">
                    <circle class="trk" cx="60" cy="60" r="54"></circle>
                    <circle class="val" cx="60" cy="60" r="54" style="stroke:{ring};--off:{offset:.1f};"></circle>
                </svg>
                <div class="txt"><b>{r['confidence'] * 100:.1f}%</b><small>model confidence</small></div>
            </div>
            """)

    st.write("")

    # ---- danger / doctor / treatment ----
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(key="card_danger"):
            segs = "".join(
                f'<i class="{"on " + tone if i < g["level"] else ""}" style="animation-delay:{0.15 * i:.2f}s"></i>'
                for i in range(5)
            )
            html(f"""
            <div class="card-h"><span class="ic">&#9888;</span><h4>How serious is it?</h4></div>
            <div class="meter">{segs}</div>
            <p class="lead">{g['level_name']}</p>
            <p>{g['danger']}</p>
            """)

    with c2:
        with st.container(key="card_doctor"):
            pts = "".join(f"<li>{p}</li>" for p in g["doctor_points"])
            html(f"""
            <div class="card-h"><span class="ic">&#129658;</span><h4>Do you need a doctor?</h4></div>
            <p class="lead">{g['doctor_head']}</p>
            <p><b>{g['doctor_when']}</b></p>
            <ul class="list">{pts}</ul>
            """)

    with c3:
        with st.container(key="card_treatment"):
            items = "".join(
                f'<div class="tx"><span class="n"></span><p><b>{t}</b>{d}</p></div>' for t, d in g["treatment"]
            )
            html(f"""
            <div class="card-h"><span class="ic">&#128138;</span><h4>{g['treatment_head']}</h4></div>
            {items}
            """)

    st.write("")

    # ---- probabilities + next steps ----
    c4, c5 = st.columns(2, gap="medium")
    with c4:
        with st.container(key="card_probs"):
            b, m = r["benign_p"] * 100, r["malignant_p"] * 100
            b_col = "#22A559" if r["benign_p"] >= r["malignant_p"] else "#94A3B8"
            m_col = "#DC2626" if r["malignant_p"] > r["benign_p"] else "#94A3B8"
            html(f"""
            <div class="card-h"><span class="ic">&#128202;</span><h4>Probability breakdown</h4></div>
            <div class="row"><span>Benign</span><span>{b:.1f}%</span></div>
            <div class="bar"><div style="--w:{b:.1f}%;background:{b_col};"></div></div>
            <div class="row"><span>Malignant</span><span>{m:.1f}%</span></div>
            <div class="bar"><div style="--w:{m:.1f}%;background:{m_col};"></div></div>
            <p style="font-size:.88rem;">Decision threshold used: <b>{r['threshold']:.2f}</b></p>
            """)
    with c5:
        with st.container(key="card_next"):
            nxt = "".join(f"<li>{n}</li>" for n in g["next"])
            html(f"""
            <div class="card-h"><span class="ic">&#9989;</span><h4>What to do next</h4></div>
            <ul class="list">{nxt}</ul>
            <div class="callout {tone}">AI results can be wrong in both directions. If you are worried about a spot,
            see a doctor even when this result looks reassuring.</div>
            """)

    st.write("")

    # ---- ABCDE ----
    with st.container(key="card_abcde"):
        cells = "".join(
            f'<div><span class="l">{a}</span><b>{t}</b><span>{d}</span></div>' for a, t, d in ABCDE
        )
        html(f"""
        <div class="card-h"><span class="ic">&#128064;</span><h4>Warning signs to watch for (ABCDE rule)</h4></div>
        <div class="abcde">{cells}</div>
        <p style="margin-top:1rem;">If a spot shows <b>any</b> of these signs, book a skin check.
        Also seek care for sores that do not heal, or a spot that bleeds, itches, or looks different from your other moles.</p>
        """)

    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.button("Analyze another image", key="again", type="primary", on_click=go, args=("detect",))

    disclaimer()


# =============================================================================
# 8. ROUTER
# =============================================================================

if ss.page == "results":
    page_results()
else:
    page_detect()
