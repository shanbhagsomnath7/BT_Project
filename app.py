import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1a73e8;
        border-radius: 20px;
        border: 1px solid #1a73e8;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #1a73e8;
    }
    h1 {
        font-family: 'Arial Black', sans-serif;
        color: #90EE90;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Functions ---
def setup_database():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, timestamp DATETIME, result TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

def log_prediction(result, confidence):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (timestamp, result, confidence) VALUES (?, ?, ?)", (datetime.now(), result, confidence))
    conn.commit()
    conn.close()

setup_database()

# --- Preprocessing & Model Loading ---
def preprocess_image(image):
    img_array = np.array(image)
    resized_array = cv2.resize(img_array, (224, 224))
    normalized_array = resized_array / 255.0
    expanded_array = np.expand_dims(normalized_array, axis=0)
    return expanded_array

@st.cache_resource
def load_ai_model():
    model = load_model('brain_tumor_model.h5')
    return model
model = load_ai_model()

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Doctor's Portal", "Admin Dashboard"])

# --- DOCTOR'S PORTAL PAGE ---
if page == "Doctor's Portal":
    st.title("Doctor's Portal: Brain Tumor Detection AI")
    st.write("Upload an MRI scan to get a prediction from the AI model.")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded MRI.', use_column_width=True)
        with col2:
            if st.button('Predict'):
                # CORRECTED INDENTATION HERE
                with st.spinner('The AI is thinking...'):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    confidence = float(prediction[0][0])

                    if confidence > 0.5:
                        result_text = "Tumor Detected"
                        st.error(f"**Result:** {result_text} (Confidence: {confidence*100:.2f}%)")
                        log_prediction(result_text, confidence)
                    else:
                        result_text = "No Tumor Detected"
                        st.success(f"**Result:** {result_text} (Confidence: {(1-confidence)*100:.2f}%)")
                        log_prediction(result_text, 1 - confidence)

# --- ADMIN DASHBOARD PAGE ---
elif page == "Admin Dashboard":
    st.title("Admin Dashboard 📊")
    st.write("This page provides analytics on the AI model's usage and results.")

    with st.form("password_form"):
        password = st.text_input("Enter password", type="password")
        submitted = st.form_submit_button("Enter")
    
    if submitted:
        if password == "admin123":
            st.success("Access Granted")
            try:
                conn = sqlite3.connect('predictions.db')
                df = pd.read_sql_query("SELECT * FROM predictions", conn)
                conn.close()
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date

                tab1, tab2, tab3 = st.tabs(["📈 Key Metrics", "🗃️ Prediction History", "📊 Results Breakdown"])

                with tab1:
                    st.header("Key Metrics")
                    total_scans = len(df)
                    positive_detections = len(df[df['result'] == 'Tumor Detected'])
                    kpi1, kpi2 = st.columns(2)
                    kpi1.metric("Total Scans Analyzed", total_scans)
                    if total_scans > 0:
                        kpi2.metric("Positive Detection Rate", f"{(positive_detections/total_scans)*100:.2f}%")
                    else:
                        kpi2.metric("Positive Detection Rate", "0.00%")
                    
                    st.header("Prediction Trend Over Time")
                    trend_data = df.groupby(['date', 'result']).size().reset_index(name='count')
                    fig = px.line(trend_data, x='date', y='count', color='result', title='Daily Prediction Volume', markers=True)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.header("Prediction History")
                    st.dataframe(df.sort_values(by='timestamp', ascending=False))

                with tab3:
                    st.header("Results Breakdown")
                    if not df.empty:
                        chart_data = df['result'].value_counts()
                        st.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Database error: {e}")
        elif password:
            st.error("Incorrect password.")
