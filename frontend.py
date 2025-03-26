import streamlit as st
import requests

# Set Streamlit app title
st.title("🎓 Student Depression Risk Predictor")

# Input fields
academic_pressure = st.slider("Academic Pressure (0-5)", 0.0, 5.0, 2.5)
work_pressure = st.slider("Work Pressure (0-5)", 0.0, 5.0, 2.5)
study_satisfaction = st.slider("Study Satisfaction (0-5)", 0.0, 5.0, 2.5)

# Sleep duration dropdown
sleep_durations = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
sleep_duration = st.selectbox("Sleep Duration", sleep_durations)

financial_stress = st.slider("Financial Stress (0-5)", 0.0, 5.0, 2.5)

# Predict button
if st.button("Predict"):
    data = {
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_duration,
        "Financial Stress": financial_stress,
    }

    # Send request to Flask API
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error("Error in prediction. Please try again.")
