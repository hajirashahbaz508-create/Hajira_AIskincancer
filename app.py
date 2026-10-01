import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("best_model.keras", compile=False)
    with open("model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_assets()
classes = info["classes"]
img_size = info["img_size"]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    if info["preprocess"] == "resnet50":
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return arr

st.title("🩺 Skin Cancer Detection")
st.caption(f"Model: {info['model_name']} | Classes: {', '.join(classes)}")
st.warning("Educational demo only — NOT a medical diagnosis. Consult a dermatologist.")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)
    with st.spinner("Analysing..."):
        probs = model.predict(preprocess(image), verbose=0)[0]
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {classes[top].upper()}")
    st.metric("Confidence", f"{probs[top] * 100:.2f}%")
    st.bar_chart({c: float(p) for c, p in zip(classes, probs)})
