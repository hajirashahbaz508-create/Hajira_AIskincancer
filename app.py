import os
import json
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Cancer AI Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Configuration & File Setup
# ---------------------------------------------------------
MODEL_PATH = "best_model.keras"
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"  # Replace with your actual Google Drive File ID
INFO_PATH = "model_info.json"

@st.cache_resource
def load_assets():
    """Downloads model if missing and loads all required assets."""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    with open(INFO_PATH, "r") as f:
        info = json.load(f)
        
    return model, info

# Load model and metadata
model, info = load_assets()
classes = info.get("classes", [])
img_size = info.get("img_size", 224)

# ---------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes and formats image for model inference."""
    img_resized = image.convert("RGB").resize((img_size, img_size))
    img_array = np.asarray(img_resized, dtype=np.float32)[None, ...]
    
    if info.get("preprocess") == "resnet50":
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
    return img_array

# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------
st.sidebar.title("📌 Model Info")
st.sidebar.markdown(f"**Architecture:** {info.get('model_name', 'CNN')}")
st.sidebar.markdown(f"**Classes ({len(classes)}):** {', '.join(classes)}")

st.title("🔬 Skin Cancer Detection")
st.caption(f"Model: {info.get('model_name', 'Deep Learning')} | Input Size: {img_size}x{img_size}")
st.warning("⚠️️ **Disclaimer:** Educational demo only — NOT a medical diagnosis. Consult a dermatologist.")

uploaded_file = st.file_uploader("Upload a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        with st.spinner("Analyzing image..."):
            processed_tensor = preprocess_image(raw_image)
            predictions = model.predict(processed_tensor, verbose=0)[0]
            top_class_idx = int(np.argmax(predictions))
            confidence_score = float(predictions[top_class_idx]) * 100

            st.subheader("Results")
            st.write(f"**Prediction:** `{classes[top_class_idx].upper()}`")
            st.metric(label="Confidence", value=f"{confidence_score:.2f}%")
