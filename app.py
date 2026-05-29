import streamlit as st
from src.predict import run_inference

st.set_page_config(page_title="CardioCare AI", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🫀 CardioCare Diagnostic Engine")
st.markdown("Input patient telemetry to assess cardiovascular risk probability.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.selectbox("Biological Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=6.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

if st.button("Run Diagnostic Scan"):
    user_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    prediction, confidence = run_inference(user_data)
    
    if prediction == 1:
        st.error(f"<h3 style='color: white;'>High Risk Detected</h3><h1 style='color: white;'>Confidence: {confidence:.1f}%</h1>", unsafe_allow_html=True)
    else:
        st.success(f"<h3 style='color: white;'>Low Risk Detected</h3><h1 style='color: white;'>Confidence: {confidence:.1f}%</h1>", unsafe_allow_html=True)