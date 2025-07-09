import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("Modelenv.v1.h5")

model = load_trained_model()

# Define classes
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Title
st.title("🌍 Satellite Image Classifier")
st.markdown("Upload a satellite image to classify it as **Cloudy, Desert, Green Area, or Water**.")

# File uploader
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

# On file upload
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

