import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import model_from_json, Sequential

# Register the Sequential class with TensorFlow Keras
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'Sequential': Sequential})

# Function to load the emotion detection model
@st.cache_resource
def load_model():
    try:
        # Open and read the model JSON file
        with open("facialemotionmodel.json", "r") as json_file:
            model_json = json_file.read()
        
        # Load the model from JSON
        model = model_from_json(model_json)
        
        # Load weights into the model
        model.load_weights("facialemotionmodel.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the face cascade
@st.cache_resource
def load_face_cascade():
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to detect emotions in a frame
def detect_emotions_in_frame(frame, model, face_cascade):
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame for faster processing
    resized_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    
    faces = face_cascade.detectMultiScale(resized_frame, 1.3, 5)
    for (p, q, r, s) in faces:
        face_image = resized_frame[q:q + s, p:p + r]
        cv2.rectangle(frame, (2*p, 2*q), (2*(p + r), 2*(q + s)), (255, 0, 0), 2)
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        pred = model.predict(img, verbose=0)  # verbose=0 to suppress warnings
        prediction_label = labels[pred.argmax()]
        cv2.putText(frame, '%s' % (prediction_label), (2*p - 10, 2*q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
    return frame

def main():
    st.title("Facial Emotion Recognition on Video")

    model = load_model()
    if model is None:
        return

    face_cascade = load_face_cascade()

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        file_bytes = uploaded_file.read()

        if uploaded_file.type.startswith('video'):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(file_bytes)
            temp_file.close()  # Close the file so it can be read by VideoCapture

            cap = cv2.VideoCapture(temp_file.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            stframe = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every alternate frame for faster display
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 0:
                    detected_frame = detect_emotions_in_frame(frame, model, face_cascade)
                    stframe.image(detected_frame, channels="BGR")

            os.unlink(temp_file.name)
            cap.release()

    st.text("Upload a video file to start facial emotion detection.")

if __name__ == "__main__":
    main()
