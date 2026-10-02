import os
import json
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Download best_model.keras from Google Drive if it doesn't exist locally
model_path = "best_model.keras"
file_id = "16tlVBG4-pbqmnWCUEaHCxtNCmQ_Y5lnS"  # Replace with your actual Google Drive File ID

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🔬", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(model_path, compile=False)
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_assets()
classes = info["classes"]
img_size = info["img_size"]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    if info.get("preprocess") == "resnet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

st.title("🔬 Skin Cancer Detection")
st.caption(f"Model: {info['model_name']} | Classes: {', '.join(classes)}")
st.warning("Educational demo only - NOT a medical diagnosis. Consult a dermatologist.")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing..."):
        probs = model.predict(preprocess(image), verbose=0)[0]
        top = int(np.argmax(probs))
        st.subheader(f"Prediction: {classes[top].upper()}")
        st.metric("Confidence", f"{probs[top] * 100:.2f}%")
