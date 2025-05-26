import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_disease_model_8_classes.h5')

model = load_model()

# Your class labels in the same order as training
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

def preprocess_image_pil(pil_img):
    pil_img = pil_img.resize((224, 224))
    img_array = np.array(pil_img)
    if img_array.shape[-1] == 4:  # Handle PNG with alpha
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

st.title("Skin Disease Detection")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption='Uploaded Image', use_container_width=True)
    st.write("")

    st.write("Classifying...")
    img_array = preprocess_image_pil(image_pil)
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence*100:.2f}%")