
import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# LOAD MODEL AND SCALER
@st.cache_resource
def load_model():
    model = joblib.load("models/best_heart_disease_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# LOAD TRAINING FEATURE COLUMNS
feature_columns = pd.read_csv("dataset/X_features.csv").columns.tolist()

# APPLICATION TITLE
st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning Based Patient Risk Analysis")

st.info(
    "This application is for educational purposes only. "
    "It cannot provide a medical diagnosis."
)

# PATIENT INPUT FORM
st.header("Enter Patient Information")

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: f"Type {x}"
        )

        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

        restecg = st.selectbox("Resting ECG", [0, 1, 2])

    with col2:
        thalach = st.number_input("Maximum Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0)
        slope = st.selectbox("ST Slope", [1, 2, 3])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [3, 6, 7])

    submitted = st.form_submit_button("Predict Heart Disease")

# PREDICTION
if submitted:

    # CREATE ORIGINAL FEATURE DICTIONARY
    patient = {
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

    # CREATE DATAFRAME
    patient_df = pd.DataFrame([patient])

    # APPLY SAME ONE-HOT ENCODING
    patient_encoded = pd.get_dummies(
        patient_df,
        columns=[
            "sex", "cp", "fbs", "restecg",
            "exang", "slope", "ca", "thal"
        ],
        dtype=int
    )

    # ALIGN FEATURES WITH TRAINING DATA
    patient_encoded = patient_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # SCALE INPUT
    patient_scaled = scaler.transform(patient_encoded)

    # PREDICT
    prediction = model.predict(patient_scaled)[0]

    st.divider()
    st.header("Prediction Result")

    if prediction == 1:
        st.error("Model Prediction: Heart Disease")
    else:
        st.success("Model Prediction: No Heart Disease")

    # SHOW PROBABILITY WHEN AVAILABLE
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(patient_scaled)[0][1]
        st.metric(
            "Estimated Model Probability",
            f"{probability * 100:.2f}%"
        )

    st.caption(
        "The result is a machine-learning estimate, not a diagnosis "
        "or a clinically validated risk score. Please consult a doctor."
    )
