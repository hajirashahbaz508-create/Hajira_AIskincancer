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
# 2. CLINICAL GUIDANCE CONTENT
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
# 3. THEME  (light palette: mist background, navy ink, indigo buttons)
# =============================================================================
# Palette
#   page        #F4F6FE -> #E9F4FA   soft mist
#   ink         #16213E               deep navy (text, ~14:1 on white)
#   muted       #4A5578               secondary text (~7:1)
#   primary     #4338CA               indigo buttons, white text (~8:1)
#   primary-dk  #3730A3               hover
#   aqua        #0EA5A4               scan line / highlights
#   good        #15803D   warn #B45309   danger #B91C1C

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
  --ink:#16213E; --muted:#4A5578; --primary:#4338CA; --primary-dk:#3730A3;
  --aqua:#0EA5A4; --line:rgba(67,56,202,.16);
  --good:#15803D; --good-bg:#E8F7EE;
  --warn:#B45309; --warn-bg:#FEF3E2;
  --danger:#B91C1C; --danger-bg:#FDECEC;
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
.block-container{max-width:1180px; padding:1.2rem 1.5rem 3rem 1.5rem;}
h1,h2,h3,h4{font-family:'Sora',sans-serif; color:var(--ink); letter-spacing:-.3px;}
p,li,label,span,small{color:var(--muted);}
b,strong{color:var(--ink);}

/* ---------- ambient background: drifting soft orbs (decorative only) ---------- */
.orb{position:fixed; border-radius:50%; filter:blur(90px); pointer-events:none; z-index:0;
     animation:drift 24s ease-in-out infinite alternate;}
.orb.o1{width:420px;height:420px;top:-120px;left:-120px;background:#A5B4FC;opacity:.45;}
.orb.o2{width:380px;height:380px;bottom:-120px;right:-100px;background:#99F6E4;opacity:.45;animation-delay:-8s;}
.orb.o3{width:260px;height:260px;top:48%;left:55%;background:#FBCFE8;opacity:.30;animation-delay:-15s;}
@keyframes drift{0%{transform:translate(0,0) scale(1);}50%{transform:translate(60px,40px) scale(1.1);}100%{transform:translate(-40px,70px) scale(.95);}}
.dust{position:fixed; top:0; left:0; width:3px; height:3px; border-radius:50%; z-index:0; pointer-events:none;
  background:transparent; animation:twinkle 4s ease-in-out infinite;
  box-shadow: 8vw 12vh #818CF8, 22vw 70vh #2DD4BF, 37vw 22vh #818CF8, 52vw 85vh #2DD4BF,
              66vw 14vh #818CF8, 79vw 58vh #2DD4BF, 91vw 33vh #818CF8, 14vw 44vh #2DD4BF,
              45vw 52vh #818CF8, 84vw 90vh #2DD4BF, 60vw 40vh #818CF8, 30vw 92vh #2DD4BF;}
@keyframes twinkle{0%,100%{opacity:.2;transform:scale(.9);}50%{opacity:.9;transform:scale(1.2);}}

/* ---------- glass cards (Streamlit containers with key="card_*") ---------- */
[class*="st-key-card"]{
  position:relative; z-index:1;
  background:rgba(255,255,255,.78);
  backdrop-filter:blur(14px) saturate(1.3); -webkit-backdrop-filter:blur(14px) saturate(1.3);
  border:1px solid var(--line); border-radius:22px; padding:1.5rem 1.6rem;
  box-shadow:0 10px 34px rgba(40,50,120,.10), inset 0 1px 0 #fff;
  animation:rise .6s cubic-bezier(.22,1,.36,1) both;
}
@keyframes rise{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:none;}}

/* ---------- top bar ---------- */
.brand{display:flex;align-items:center;gap:.7rem;font-family:'Sora',sans-serif;font-weight:800;font-size:1.2rem;color:var(--ink);padding-top:.35rem;}
.brand .logo{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;
  background:linear-gradient(135deg,#4338CA,#0EA5A4);box-shadow:0 6px 18px rgba(67,56,202,.35);position:relative;}
.brand .logo::after{content:"";position:absolute;inset:-5px;border-radius:15px;border:2px solid rgba(67,56,202,.35);animation:ring 2.4s ease-out infinite;}
.brand .logo i{width:14px;height:14px;border-radius:50%;background:#fff;display:block;}
@keyframes ring{0%{transform:scale(.9);opacity:.9;}100%{transform:scale(1.35);opacity:0;}}

/* ---------- hero ---------- */
.hero{position:relative;z-index:1;text-align:center;padding:2.6rem 1rem 1.6rem;}
.hero h1{font-size:clamp(2rem,5.2vw,3.5rem);font-weight:800;margin:0 0 .6rem;line-height:1.08;}
.hero h1 .shine{background:linear-gradient(90deg,#4338CA,#0EA5A4,#4338CA);background-size:250% auto;
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 6s linear infinite;}
@keyframes shine{to{background-position:250% center;}}
.hero p{max-width:640px;margin:0 auto;font-size:clamp(.98rem,2.4vw,1.15rem);line-height:1.6;}
.pill{display:inline-block;margin-top:1.1rem;padding:.45rem 1.1rem;border-radius:999px;font-size:.82rem;font-weight:600;
  color:#7C2D12;background:#FFEDD5;border:1px solid #FDBA74;}

/* ---------- stepper ---------- */
.steps{position:relative;z-index:1;display:flex;justify-content:center;align-items:center;gap:.6rem;margin:.4rem 0 1.4rem;flex-wrap:wrap;}
.step{display:flex;align-items:center;gap:.5rem;padding:.45rem 1rem .45rem .5rem;border-radius:999px;background:rgba(255,255,255,.7);
  border:1px solid var(--line);font-weight:600;font-size:.9rem;color:var(--muted);transition:all .4s ease;}
.step b{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:.8rem;background:#E0E4F8;color:var(--primary);}
.step.on{color:var(--ink);border-color:var(--primary);box-shadow:0 4px 16px rgba(67,56,202,.18);}
.step.on b{background:var(--primary);color:#fff;}
.step.done b{background:var(--good);color:#fff;}
.step-line{width:28px;height:2px;background:var(--line);}

/* ---------- uploader ---------- */
[data-testid="stFileUploader"] section{
  background:#F5F6FF; border:2px dashed rgba(67,56,202,.45); border-radius:18px; padding:1.4rem; transition:all .3s ease;}
[data-testid="stFileUploader"] section:hover{border-color:var(--primary);background:#EEF0FF;box-shadow:0 0 0 5px rgba(67,56,202,.10);}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span{color:var(--muted) !important;}
[data-testid="stFileUploader"] button{
  background:#fff !important;color:var(--primary) !important;border:2px solid var(--primary) !important;border-radius:12px !important;font-weight:700 !important;}
[data-testid="stFileUploader"] button:hover{background:var(--primary) !important;color:#fff !important;}

/* ---------- BUTTONS: dark fill + white text on light page; secondary = outlined ---------- */
.stButton>button{
  width:100%; border-radius:14px; font-weight:700; font-size:1.02rem; padding:.85rem 1.2rem;
  transition:transform .2s ease, box-shadow .2s ease, background .2s ease;
}
.stButton>button p{font-weight:700; margin:0;}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(120deg,#4338CA,#3730A3); color:#fff !important; border:2px solid #3730A3;
  box-shadow:0 8px 22px rgba(67,56,202,.35);
}
.stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p{color:#fff !important;}
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
.stButton>button:disabled{background:#D9DCEB !important;border-color:#C5C9DE !important;box-shadow:none !important;}
.stButton>button:disabled p{color:#5B6385 !important;}
.stButton>button:focus-visible{outline:3px solid #0EA5A4; outline-offset:3px;}

/* ---------- slider ---------- */
[data-baseweb="slider"] div[role="slider"]{background:var(--primary) !important;border-color:#fff !important;box-shadow:0 0 0 4px rgba(67,56,202,.25) !important;}
[data-testid="stSliderThumbValue"]{color:var(--primary) !important;font-weight:700;}
.stSlider label{color:var(--ink) !important;}

/* ---------- image with scan animation ---------- */
.imgbox{position:relative;border-radius:18px;overflow:hidden;border:1px solid var(--line);box-shadow:0 10px 28px rgba(40,50,120,.18);background:#EEF0FF;}
.imgbox img{display:block;width:100%;height:auto;max-height:420px;object-fit:cover;}
.imgbox.scan::after{content:"";position:absolute;left:0;right:0;height:56px;top:-60px;
  background:linear-gradient(180deg,rgba(14,165,164,0),rgba(14,165,164,.45),rgba(14,165,164,0));
  border-bottom:2px solid #0EA5A4;animation:scan 3.2s ease-in-out infinite;}
@keyframes scan{0%{top:-60px;}100%{top:100%;}}
.corner{position:absolute;width:26px;height:26px;border:3px solid #fff;filter:drop-shadow(0 1px 3px rgba(0,0,0,.4));}
.corner.tl{top:12px;left:12px;border-right:0;border-bottom:0;border-radius:8px 0 0 0;}
.corner.tr{top:12px;right:12px;border-left:0;border-bottom:0;border-radius:0 8px 0 0;}
.corner.bl{bottom:12px;left:12px;border-right:0;border-top:0;border-radius:0 0 0 8px;}
.corner.br{bottom:12px;right:12px;border-left:0;border-top:0;border-radius:0 0 8px 0;}
.empty{display:grid;place-items:center;text-align:center;min-height:260px;border-radius:18px;border:2px dashed rgba(67,56,202,.25);background:#F7F8FF;padding:1rem;}
.empty .dot{width:54px;height:54px;border-radius:50%;margin:0 auto .8rem;background:radial-gradient(circle,#818CF8,#4338CA);animation:breathe 2.6s ease-in-out infinite;}
@keyframes breathe{0%,100%{transform:scale(.9);box-shadow:0 0 0 0 rgba(67,56,202,.4);}50%{transform:scale(1.05);box-shadow:0 0 0 18px rgba(67,56,202,0);}}

/* ---------- analyzing overlay ---------- */
.analyzing{position:relative;z-index:1;text-align:center;padding:2.4rem 1rem;}
.helix{display:flex;justify-content:center;gap:7px;height:70px;align-items:center;margin-bottom:1rem;}
.helix span{width:10px;border-radius:6px;background:linear-gradient(180deg,#4338CA,#0EA5A4);animation:wave 1.1s ease-in-out infinite;}
.helix span:nth-child(1){animation-delay:0s}.helix span:nth-child(2){animation-delay:.1s}.helix span:nth-child(3){animation-delay:.2s}
.helix span:nth-child(4){animation-delay:.3s}.helix span:nth-child(5){animation-delay:.4s}.helix span:nth-child(6){animation-delay:.5s}
.helix span:nth-child(7){animation-delay:.6s}
@keyframes wave{0%,100%{height:14px;opacity:.5;}50%{height:64px;opacity:1;}}
.analyzing h3{margin:0 0 .3rem;}

/* ---------- result visuals ---------- */
.badge{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1.3rem;border-radius:999px;font-family:'Sora',sans-serif;font-weight:800;font-size:1.15rem;
  animation:pop .6s cubic-bezier(.34,1.56,.64,1) both;}
.badge::before{content:"";width:11px;height:11px;border-radius:50%;background:currentColor;animation:pulse 1.8s ease-out infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 currentColor;}70%{box-shadow:0 0 0 10px transparent;}100%{box-shadow:0 0 0 0 transparent;}}
@keyframes pop{0%{opacity:0;transform:scale(.8);}100%{opacity:1;transform:scale(1);}}
.badge.good{background:var(--good-bg);color:var(--good);border:2px solid #86D5A3;}
.badge.warn{background:var(--warn-bg);color:var(--warn);border:2px solid #F5C98A;}
.badge.danger{background:var(--danger-bg);color:var(--danger);border:2px solid #F4A3A3;}
.res-title{font-size:clamp(1.5rem,3.6vw,2.2rem);font-weight:800;margin:.9rem 0 .4rem;}
.res-sum{font-size:1.05rem;line-height:1.65;max-width:560px;}

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

.card-h{display:flex;align-items:center;gap:.6rem;margin:0 0 .7rem;}
.card-h .ic{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;font-size:1.05rem;background:#EEF0FF;}
.card-h h4{margin:0;font-size:1.1rem;}
.lead{font-weight:700;color:var(--ink);font-size:1.05rem;margin:.2rem 0;}
.list{margin:.4rem 0 0;padding-left:1.15rem;}.list li{margin:.4rem 0;line-height:1.55;}
.tx{display:flex;gap:.7rem;align-items:flex-start;padding:.6rem .7rem;border-radius:12px;background:#F5F6FF;margin:.45rem 0;}
.tx .n{min-width:8px;height:8px;border-radius:50%;background:var(--primary);margin-top:.5rem;}
.tx p{margin:0;line-height:1.5;}.tx p b{display:block;}
.callout{padding:.9rem 1.1rem;border-radius:14px;margin-top:1rem;font-weight:500;line-height:1.55;}
.callout.good{background:var(--good-bg);color:#14532D;border-left:5px solid var(--good);}
.callout.warn{background:var(--warn-bg);color:#78350F;border-left:5px solid var(--warn);}
.callout.danger{background:var(--danger-bg);color:#7F1D1D;border-left:5px solid var(--danger);}
.abcde{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;}
.abcde div{background:#F5F6FF;border-radius:16px;padding:1rem .8rem;text-align:center;border:1px solid var(--line);transition:transform .25s ease, box-shadow .25s ease;}
.abcde div:hover{transform:translateY(-4px);box-shadow:0 10px 22px rgba(67,56,202,.15);}
.abcde .l{font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:800;color:var(--primary);}
.abcde b{display:block;margin:.1rem 0 .3rem;}.abcde span{font-size:.86rem;line-height:1.4;display:block;}
.disclaimer{position:relative;z-index:1;text-align:center;font-size:.85rem;line-height:1.6;margin-top:1.4rem;padding:1rem 1.2rem;border-radius:16px;
  background:rgba(255,255,255,.7);border:1px solid var(--line);}

@media (max-width:900px){
  .abcde{grid-template-columns:repeat(2,1fr);}
  .block-container{padding:1rem .9rem 2.5rem;}
  [class*="st-key-card"]{padding:1.1rem;border-radius:18px;}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01ms !important;animation-iteration-count:1 !important;transition:none !important;}}
</style>
<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div><div class="dust"></div>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# =============================================================================
# 4. SESSION STATE & NAVIGATION
# =============================================================================

ss = st.session_state
ss.setdefault("page", "detect")
ss.setdefault("result", None)


def go(page: str):
    ss.page = page


def top_bar():
    c1, c2, c3 = st.columns([4, 1.3, 1.3], gap="small", vertical_alignment="center")
    with c1:
        html('<div class="brand"><span class="logo"><i></i></span>DermaScan</div>')
    with c2:
        st.button("Detect", key="nav_detect", on_click=go, args=("detect",),
                  type="primary" if ss.page == "detect" else "secondary")
    with c3:
        st.button("Results", key="nav_results", on_click=go, args=("results",),
                  type="primary" if ss.page == "results" else "secondary",
                  disabled=ss.result is None)


def stepper(active: int):
    items = ["Upload image", "Analyze", "Read results"]
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
# 5. PAGE 1 - DETECT
# =============================================================================

def page_detect():
    top_bar()
    html("""
    <div class="hero">
        <h1>Check a skin spot<br><span class="shine">in three simple steps</span></h1>
        <p>Upload a close-up photo of a skin lesion and our AI model will estimate whether it looks
        benign or malignant, then explain what to do next.</p>
        <span class="pill">Research and education tool - not a medical diagnosis</span>
    </div>
    """)

    model = load_model()
    uploaded = ss.get("uploader")
    stepper(2 if uploaded is not None else 1)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        with st.container(key="card_upload"):
            html('<div class="card-h"><span class="ic">&#128247;</span><h4>1. Upload your photo</h4></div>')
            uploaded = st.file_uploader(
                "Upload a lesion image (jpg, jpeg or png)", type=["jpg", "jpeg", "png"],
                key="uploader", label_visibility="collapsed",
            )
            html("""
            <ul class="list">
                <li>Sharp, in-focus close-up with the spot filling most of the frame</li>
                <li>Bright, even light - avoid shadows and flash glare</li>
                <li>No filters, and try to keep hair or rulers out of the spot</li>
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
                    <p>Upload a photo on the left and it will appear here.</p>
                </div></div>
                """)

    with st.container(key="card_settings"):
        html('<div class="card-h"><span class="ic">&#9881;</span><h4>2. Sensitivity (optional)</h4></div>')
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
        slot.markdown(H("""
        <div class="analyzing" style="position:relative;z-index:1;">
            <div class="helix"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
            <h3>Analyzing your image...</h3>
            <p>Scanning texture, colour, and border patterns. This takes a few seconds.</p>
        </div>
        """), unsafe_allow_html=True)

        image = load_image(uploaded)
        batch = preprocess_image(image)
        _, confidence, probs = predict(model, batch)
        time.sleep(0.8)

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
# 6. PAGE 2 - RESULTS
# =============================================================================

def page_results():
    top_bar()
    r = ss.result
    if r is None:
        with st.container(key="card_empty"):
            html('<div class="empty"><div><div class="dot"></div><p class="lead">No results yet</p>'
                 '<p>Go to the Detect page, upload an image, and press Analyze.</p></div></div>')
            st.button("Go to Detect", key="empty_go", on_click=go, args=("detect",), type="primary")
        disclaimer()
        return

    g = GUIDANCE[r["key"]]
    tone = g["tone"]
    ring = {"good": "#22A559", "warn": "#E8890C", "danger": "#DC2626"}[tone]
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
            <span class="badge {tone}">{badge_text}</span>
            <div class="res-title">{g['title']}</div>
            <p class="res-sum">{g['summary']}</p>
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
# 7. ROUTER
# =============================================================================

if ss.page == "results":
    page_results()
else:
    page_detect()
