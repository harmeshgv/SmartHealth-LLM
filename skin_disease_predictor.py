import streamlit as st
st.set_page_config(page_title="Skin Disease Predictor", layout="centered")  # 🔺 First Streamlit command

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model once when the app starts
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_disease_model_8_classes.h5')

model = load_model()

# Class labels
CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

def preprocess_image(uploaded_image):
    img = uploaded_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_skin_disease(img):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)
    return predicted_class, confidence

# --- Streamlit UI ---

st.title("🩺 Skin Disease Prediction App")
st.write("Upload an image of a skin condition, and the model will predict the type of disease.")

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_skin_disease(image_pil)
            st.success(f"**Prediction:** {prediction}")
            st.info(f"**Confidence:** {confidence:.2f}")
