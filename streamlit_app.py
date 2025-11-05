import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import pyttsx3
from mediapipe_utils import extract_landmarks

# Set page config (only set once)
st.set_page_config(page_title="Sign Language Recognition", layout="centered")

# CSS Styling
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Segoe UI', sans-serif;
}

h1 {
    color: #003366;
    text-align: center;
}

.stApp {
    background-image: url('https://www.transparenttextures.com/patterns/connected.png');
    background-size: cover;
    background-repeat: no-repeat;
}

.prediction {
    font-size: 48px;
    font-weight: bold;
    color: #00b894;
    text-align: center;
    padding: 10px;
    background-color: #dfe6e9;
    border-radius: 15px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.2);
}

.footer {
    font-size: 16px;
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>🤟 Sign Language to Text & Voice</h1>", unsafe_allow_html=True)

# Load model and labels
model = tf.keras.models.load_model("trained_model/sign_model.keras")
with open("trained_model/labels.txt") as f:
    labels = f.read().splitlines()

# Init engine and camera
engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

# UI placeholders
run = st.button("Start Camera")
frame_placeholder = st.empty()
output_text = st.empty()

# Track previous prediction
previous_label = None

# Main logic
if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_landmarks(frame)
        if landmarks.any():
            resized_frame = cv2.resize(frame, (100, 100))
            resized_frame = resized_frame / 255.0
            prediction = model.predict(np.expand_dims(resized_frame, axis=0))[0]
            pred_index = np.argmax(prediction)
            pred_label = labels[pred_index]

            # Speak only if label changes
            if pred_label != previous_label:
                output_text.markdown(f"<div class='prediction'>Predicted: {pred_label}</div>", unsafe_allow_html=True)
                engine.say(pred_label)
                engine.runAndWait()
                previous_label = pred_label

        frame_placeholder.image(frame, channels="BGR")

# Add footer
st.markdown("<div class='footer'>Made with ❤️ by Soundarya</div>", unsafe_allow_html=True)
