import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Probability Dashboard",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Files Safely (Deployment Ready)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "heart_feature_columns.pkl"))

# Optional dataset load for analytics
data_path = os.path.join(BASE_DIR, "heart.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None

# -----------------------------
# Helper Functions
# -----------------------------
def get_risk_level(prob):
    if prob < 0.30:
        return "Low Risk", "green"
    elif prob < 0.60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def build_input_df(inputs):
    data = {col: inputs.get(col, 0) for col in feature_columns}
    return pd.DataFrame([data])

# -----------------------------
# Main Title
# -----------------------------
st.title("❤️ Heart Disease Probability Dashboard Model")
st.markdown("""
Estimate the **likelihood of heart disease** based on patient health metrics.  
This dashboard helps **doctors, students, and analysts** compare patient risk using a machine learning model.
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("🩺 Patient Health Inputs")
st.sidebar.markdown("Enter patient details below.")

age = st.sidebar.slider("Age", 20, 90, 45)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 80, 220, 120)
chol = st.sidebar.slider("Cholesterol (chol)", 100, 600, 220)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# -----------------------------
# Prepare Input Data
# -----------------------------
user_inputs = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

input_df = build_input_df(user_inputs)

# Scale the input
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction Section
# -----------------------------
if st.sidebar.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    risk_label, risk_color = get_risk_level(probability)

    st.subheader("📌 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted Class", "Heart Disease" if prediction == 1 else "No Heart Disease")

    with c2:
        st.metric("Disease Probability", f"{probability * 100:.2f}%")

    with c3:
        st.markdown(f"### Risk Level: :{risk_color}[{risk_label}]")

    # Risk meter
    st.subheader("📊 Risk Probability Meter")
    st.progress(float(probability))

    # Clinical Interpretation
    st.subheader("🧠 Clinical Interpretation")
    if probability < 0.30:
        st.success("This patient appears to be at **low risk** based on the provided health values.")
    elif probability < 0.60:
        st.warning("This patient appears to be at **moderate risk**. Further medical review is recommended.")
    else:
        st.error("This patient appears to be at **high risk**. Immediate medical evaluation is strongly recommended.")

    # Input Summary
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_df)

    # Patient Health Profile
    st.subheader("📈 Patient Health Profile")
    chart_df = pd.DataFrame({
        "Feature": ["Age", "Resting BP", "Cholesterol", "Max Heart Rate", "Oldpeak"],
        "Value": [age, trestbps, chol, thalach, oldpeak]
    })
    st.bar_chart(chart_df.set_index("Feature"))

    # Probability Breakdown
    st.subheader("📊 Prediction Probability Breakdown")
    prob_df = pd.DataFrame({
        "Outcome": ["No Heart Disease", "Heart Disease"],
        "Probability": [1 - probability, probability]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

# -----------------------------
# Dataset Analytics Section
# -----------------------------
st.markdown("---")
st.header("📚 Dataset Insights & Analytics")

if df is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Disease Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df["target"].value_counts().sort_index().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Target (0 = No Disease, 1 = Disease)")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        st.subheader("Age Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df["age"].plot(kind="hist", bins=20, ax=ax2)
        ax2.set_xlabel("Age")
        st.pyplot(fig2)

    # Cholesterol vs Target
    st.subheader("Cholesterol by Heart Disease")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="chol", by="target", ax=ax3)
    plt.suptitle("")
    ax3.set_title("Cholesterol vs Target")
    ax3.set_xlabel("Target")
    ax3.set_ylabel("Cholesterol")
    st.pyplot(fig3)

    # Resting BP vs Target
    st.subheader("Resting Blood Pressure by Heart Disease")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="trestbps", by="target", ax=ax4)
    plt.suptitle("")
    ax4.set_title("Resting BP vs Target")
    ax4.set_xlabel("Target")
    ax4.set_ylabel("Resting BP")
    st.pyplot(fig4)

    # Max Heart Rate vs Target
    st.subheader("Maximum Heart Rate by Heart Disease")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="thalach", by="target", ax=ax5)
    plt.suptitle("")
    ax5.set_title("Max Heart Rate vs Target")
    ax5.set_xlabel("Target")
    ax5.set_ylabel("thalach")
    st.pyplot(fig5)

else:
    st.info("`heart.csv` not found in the project folder. Add it to show dataset analytics.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
### ⚠️ Medical Disclaimer
This dashboard is an **educational machine learning project** and should **not replace professional medical diagnosis**.  
Always consult a qualified healthcare professional before making medical decisions.
""")