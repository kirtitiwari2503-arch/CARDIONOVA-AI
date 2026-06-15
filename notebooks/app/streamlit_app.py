import streamlit as st
import joblib
import numpy as np

# --------------------------------------------------
# CARDIONOVA - Heart Disease Risk Predictor
# Educational Machine Learning Project
# --------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="CARDIONOVA",
    page_icon="❤️",
    layout="centered"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/cardio_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# Title and Description
# --------------------------------------------------
st.title("❤️ CARDIONOVA – Heart Disease Risk Predictor")

st.markdown(
    """
    Predict cardiovascular disease risk using a trained machine learning model.

    **Disclaimer:** This is an educational project and not a medical diagnosis tool.
    """
)

st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------
st.subheader("Patient Information")

with st.form("prediction_form"):

    age = st.slider(
        "Age (Years)",
        min_value=18,
        max_value=100,
        value=40
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    height = st.slider(
        "Height (cm)",
        min_value=100,
        max_value=250,
        value=170
    )

    weight = st.slider(
        "Weight (kg)",
        min_value=30,
        max_value=200,
        value=70
    )

    st.subheader("Health Metrics")

    ap_hi = st.slider(
        "Systolic Blood Pressure (ap_hi)",
        min_value=80,
        max_value=250,
        value=120
    )

    ap_lo = st.slider(
        "Diastolic Blood Pressure (ap_lo)",
        min_value=40,
        max_value=180,
        value=80
    )

    cholesterol_text = st.selectbox(
        "Cholesterol",
        [
            "Normal",
            "Above Normal",
            "Well Above Normal"
        ]
    )

    glucose_text = st.selectbox(
        "Glucose",
        [
            "Normal",
            "Above Normal",
            "Well Above Normal"
        ]
    )

    st.subheader("Lifestyle Information")

    smoke_text = st.radio(
        "Do you smoke?",
        ["No", "Yes"]
    )

    alcohol_text = st.radio(
        "Do you consume alcohol?",
        ["No", "Yes"]
    )

    active_text = st.radio(
        "Are you physically active?",
        ["Yes", "No"]
    )

    predict_button = st.form_submit_button("🔍 Predict Risk")

# --------------------------------------------------
# Feature Encoding
# --------------------------------------------------
if predict_button:

    gender_encoded = 1 if gender == "Male" else 0

    cholesterol_map = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }

    glucose_map = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }

    cholesterol = cholesterol_map[cholesterol_text]
    glucose = glucose_map[glucose_text]

    smoke = 1 if smoke_text == "Yes" else 0
    alcohol = 1 if alcohol_text == "Yes" else 0
    active = 1 if active_text == "Yes" else 0

    # --------------------------------------------------
    # Create feature array
    # IMPORTANT:
    # Ensure feature order matches training dataset.
    # --------------------------------------------------

    patient_id = 0
    age_years = age
    bmi = weight / ((height / 100) ** 2)
    obesity = 1 if bmi >= 30 else 0

    features = np.array([[
        patient_id,
        age,
        gender_encoded,
        height,
        weight,
        ap_hi,
        ap_lo,
        cholesterol,
        glucose,
        smoke,
        alcohol,
        active,
        age_years,
        bmi,
        obesity
    ]])

    try:
        prediction = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("❤️ No Disease Risk Detected")
        else:
            st.error("⚠️ Disease Risk Detected")

        if probability is not None:
            st.metric(
                "Risk Probability",
                f"{probability * 100:.2f}%"
            )

        st.info(
            "This is an educational project and not a medical diagnosis tool."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "CARDIONOVA runs completely offline. No user data is stored, uploaded, or shared."
)