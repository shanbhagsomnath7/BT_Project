import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- Page Configuration (NEW UI CODE) ---
st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Preprocessing Function ---
def preprocess_image(image):
    img_array = np.array(image)
    resized_array = cv2.resize(img_array, (224, 224))
    normalized_array = resized_array / 255.0
    expanded_array = np.expand_dims(normalized_array, axis=0)
    return expanded_array

# --- Load the AI Model ---
@st.cache_resource
def load_ai_model():
    model = load_model('brain_tumor_model.h5')
    return model

model = load_ai_model()

# --- Main App Interface ---
st.title("Brain Tumor Detection AI 🧠")
st.write("Upload an MRI scan to get a prediction from the AI model.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
    with col2:
        if st.button('Predict'):
            with st.spinner('The AI is thinking...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                confidence = float(prediction[0][0])

                if confidence > 0.5:
                    result_text = "Tumor Detected"
                    st.error(f"**Result:** {result_text} (Confidence: {confidence*100:.2f}%)")
                else:
                    result_text = "No Tumor Detected"
                    st.success(f"**Result:** {result_text} (Confidence: {(1-confidence)*100:.2f}%)")
