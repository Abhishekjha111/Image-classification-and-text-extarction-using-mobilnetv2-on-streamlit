import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pytesseract
import os

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  # Update this path as necessary

# Load the MobileNetV2 model pre-trained on ImageNet
@st.cache_data
def load_model():
    model_path = 'mobilenetv2_local_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure the model file is in the correct directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def decode_predictions(predictions):
    try:
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error decoding predictions: {e}")
        return []

def predict(image, model):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is not None:
        try:
            predictions = model.predict(preprocessed_image)
            decoded = decode_predictions(predictions)
            return decoded
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return []

def extract_text(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Streamlit app setup
st.title("Image Classification and Text Extraction with MobileNetV2")

model = load_model()

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the image
    st.write("Classifying...")
    predictions = predict(image, model)
    if predictions:
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"Prediction {i+1}: {label} with confidence {score:.2f}")

    # Extract text from the image
    st.write("Extracting text...")
    extracted_text = extract_text(image)
    st.text_area("Extracted Text", extracted_text, height=300)
