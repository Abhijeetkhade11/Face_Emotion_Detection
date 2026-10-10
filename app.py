import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Set up the Streamlit page configuration
st.set_page_config(page_title="Face Emotion Detection", page_icon="🎭", layout="centered")

# Load model and cascade (cache to avoid reloading on every interaction)
@st.cache_resource
def load_emotion_model():
    return load_model('emotion_model.keras')

import os

def load_face_cascade():
    # Load from the local file we just downloaded to bypass venv bugs
    return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_emotion_model()
face_cascade = load_face_cascade()

# Classes matching the model's training data
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_face(face_img):
    """Preprocess the detected face to match model input requirements."""
    face = cv2.resize(face_img, (48, 48))          # Resize to match model input (48x48)
    face = face / 255.0                            # Normalize pixel values
    face = np.expand_dims(face, axis=-1)           # Add channel dimension (48,48,1)
    face = np.expand_dims(face, axis=0)            # Add batch dimension (1,48,48,1)
    return face

def predict_emotion(face_img):
    """Predict emotion from preprocessed face."""
    processed = preprocess_face(face_img)
    prediction = model.predict(processed, verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx]

# Frontend UI
st.title("🎭 Face Emotion Detection App")
st.write("Upload an image or start the live webcam to detect facial emotions instantly!")
st.markdown("---")

# Use tabs to organize the features
tab1, tab2 = st.tabs(["🖼️ Image Upload", "📷 Live Webcam"])

# ----------------- Image Upload Tab -----------------
with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image using PIL and force convert to RGB
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (relaxed parameters slightly to catch more faces)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))
        
        if len(faces) == 0:
            st.warning("No faces detected in the image. Please try another one.")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            # Process each detected face
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                label = predict_emotion(face_gray)
                
                # Draw bounding box and label on the original RGB image array
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add a filled background for text to make it readable
                cv2.rectangle(img_array, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_array, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                
            st.image(img_array, caption="Detected Emotions", use_container_width=True)
            st.success(f"Detected {len(faces)} face(s) successfully!")

# ----------------- Live Webcam Tab -----------------
with tab2:
    st.header("Live Webcam Feed")
    st.write("Click the checkbox below to turn your webcam on and off.")
    
    run = st.checkbox("Turn On Webcam")
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Please check your camera permissions.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, minNeighbors=10)
            
            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                label = predict_emotion(face_gray)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            # Convert BGR (OpenCV format) back to RGB (Streamlit format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()
    else:
        st.info("Webcam is turned off.")
