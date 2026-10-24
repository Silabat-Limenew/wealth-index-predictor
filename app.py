import streamlit as st
import pandas as pd
import joblib

# Set page title and configuration
st.set_page_config(
    page_title="Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('best_wealth_predictor_pipeline.pkl')

pipeline = load_model()

# Dictionary mapping encoded classes back to DHS wealth tiers
WEALTH_LABELS = {
    0: "Poorest",
    1: "Poorer",
    2: "Middle",
    3: "Richer",
    4: "Richest"
}

st.title("Predictor")
st.write("Select household characteristics to predict wealth category.")

# --- FORM INPUTS ---
hv201 = st.selectbox(
    "Drinking Water Source (hv201)",
    options=["11", "12", "13", "14", "21", "31", "41", "51", "61", "96"],
    format_func=lambda x: {
        "11": "Piped into dwelling",
        "12": "Piped to yard/plot",
        "13": "Public tap/standpipe",
        "14": "Piped to neighbor",
        "21": "Tube well or borehole",
        "31": "Protected well",
        "41": "Protected spring",
        "51": "Rainwater",
        "61": "Tanker truck",
        "96": "Other"
    }.get(x, x)
)

hv205 = st.selectbox(
    "Toilet Type (hv205)",
    options=["11", "12", "21", "22", "23", "31", "96"],
    format_func=lambda x: {
        "11": "Flush to piped sewer system",
        "12": "Flush to septic tank",
        "21": "Flush to pit latrine",
        "22": "Flush to somewhere else",
        "23": "Flush to unknown place/not sure",
        "31": "Ventilated Improved Pit latrine (VIP)",
        "96": "Other"
    }.get(x, x)
)

hv206 = st.selectbox(
    "Electricity Available (hv206)",
    options=["0", "1"],
    format_func=lambda x: "Yes" if x == "1" else "No"
)

hv207 = st.selectbox(
    "Radio Available (hv207)",
    options=["0", "1"],
    format_func=lambda x: "Yes" if x == "1" else "No"
)

hv208 = st.selectbox(
    "Television Available (hv208)",
    options=["0", "1"],
    format_func=lambda x: "Yes" if x == "1" else "No"
)

hv025 = st.selectbox(
    "Place of Residence (hv025)",
    options=["1", "2"],
    format_func=lambda x: "Urban" if x == "1" else "Rural"
)

hv024 = st.selectbox("Region (hv024)", options=[str(i) for i in range(1, 12)])
hv209 = st.selectbox("Refrigerator Available (hv209)", options=["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
hv210 = st.selectbox("Bicycle Available (hv210)", options=["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
hv211 = st.selectbox("Motorcycle Available (hv211)", options=["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
hv212 = st.selectbox("Car/Truck Available (hv212)", options=["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")

# --- PREDICTION ---
if st.button("Predict Wealth Index"):
    input_data = pd.DataFrame([{
        'hv201': str(hv201),
        'hv205': str(hv205),
        'hv206': str(hv206),
        'hv207': str(hv207),
        'hv208': str(hv208),
        'hv025': str(hv025),
        'hv024': str(hv024),
        'hv209': str(hv209),
        'hv210': str(hv210),
        'hv211': str(hv211),
        'hv212': str(hv212)
    }])

    pred_class = pipeline.predict(input_data)[0]
    predicted_label = WEALTH_LABELS.get(pred_class, str(pred_class))
    
    st.success(f"Predicted Wealth Category: {predicted_label}")
