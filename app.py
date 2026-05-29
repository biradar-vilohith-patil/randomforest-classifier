import streamlit as st
from src.predict import run_inference

st.set_page_config(page_title="Sleep Health AI", page_icon="🌙", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🌙 Sleep Health Assessment")
st.markdown("<p style='text-align: center; color: #a0aec0;'>Enter your daily lifestyle habits to check your risk profile for sleep disorders.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.slider("Age", min_value=18, max_value=90, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi_category = st.selectbox("Body Mass Index Category", ["Normal", "Overweight", "Obese"])
    sleep_duration = st.slider("Average Sleep Duration (Hours)", min_value=3.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    stress_level = st.slider("Current Stress Level (1-10)", min_value=1, max_value=10, value=5)
    quality_of_sleep = st.slider("Subjective Sleep Quality (1-10)", min_value=1, max_value=10, value=6)
    physical_activity = st.slider("Physical Activity (Minutes/Day)", min_value=0, max_value=120, value=45)
    daily_steps = st.slider("Daily Step Count", min_value=0, max_value=20000, value=6000, step=500)

if st.button("Analyze Sleep Profile"):
    user_data = {
        'Gender': gender,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_category,
        'Daily Steps': daily_steps
    }
    
    prediction, confidence = run_inference(user_data)
    
    if prediction == 'None':
        st.success("Healthy Sleep Profile Detected")
        st.markdown(f"<h4 style='color: #a7f3d0; text-align: center;'>Confidence: {confidence:.1f}%</h4><p style='text-align: center; color: white;'>Your lifestyle metrics indicate a low risk for clinical sleep disorders.</p>", unsafe_allow_html=True)
    elif prediction == 'Insomnia':
        st.warning("Insomnia Risk Detected")
        st.markdown(f"<h4 style='color: #fde68a; text-align: center;'>Confidence: {confidence:.1f}%</h4><p style='text-align: center; color: white;'>Your profile shows markers commonly associated with Insomnia. Consider evaluating your stress levels and screen time before bed.</p>", unsafe_allow_html=True)
    elif prediction == 'Sleep Apnea':
        st.error("Sleep Apnea Risk Detected")
        st.markdown(f"<h4 style='color: #fecaca; text-align: center;'>Confidence: {confidence:.1f}%</h4><p style='text-align: center; color: white;'>Your profile aligns with patterns of Sleep Apnea. It is highly recommended to consult a physician for a formal sleep study.</p>", unsafe_allow_html=True)