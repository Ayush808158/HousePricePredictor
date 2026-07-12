"""
House Price Predictor
----------------------
A Streamlit web app that serves a trained Linear Regression model
to predict house prices (INR) based on property features.

Run locally:
    streamlit run app.py

Required files in the same folder:
    - model.pkl              (trained sklearn model)
    - feature_columns.json   (exact column order used in training)
    - categories.json        (valid Location / Furnished values)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"
COLUMNS_PATH = BASE_DIR / "feature_columns.json"
CATEGORIES_PATH = BASE_DIR / "categories.json"


# --------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "`model.pkl` not found. Run the training notebook's "
            "save-model cell first, then place model.pkl next to app.py."
        )
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model.pkl — real error: {type(e).__name__}: {e}")
        st.write(f"File size on disk: {MODEL_PATH.stat().st_size} bytes")
        st.stop()


@st.cache_data(show_spinner=False)
def load_feature_columns():
    if not COLUMNS_PATH.exists():
        st.error("`feature_columns.json` not found next to app.py.")
        st.stop()
    with open(COLUMNS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_categories():
    if not CATEGORIES_PATH.exists():
        st.error("`categories.json` not found next to app.py.")
        st.stop()
    with open(CATEGORIES_PATH, "r") as f:
        return json.load(f)


def format_inr(amount: float) -> str:
    """Format a number as an Indian-style rupee string with Cr/Lakh."""
    if amount >= 1_00_00_000:
        return f"₹ {amount / 1_00_00_000:.2f} Cr"
    elif amount >= 1_00_000:
        return f"₹ {amount / 1_00_000:.2f} L"
    return f"₹ {amount:,.0f}"


def build_input_row(inputs: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Turn the raw user inputs into a single-row DataFrame that matches
    the exact one-hot encoded column layout the model was trained on.
    """
    row = {
        "Area_sqft": inputs["area"],
        "Bedrooms": inputs["bedrooms"],
        "Bathrooms": inputs["bathrooms"],
        "Age_years": inputs["age"],
        "Distance_to_City_Center_km": inputs["distance"],
        "Garden": int(inputs["garden"]),
        "Swimming_Pool": int(inputs["pool"]),
        "Security_System": int(inputs["security"]),
        "Gym": int(inputs["gym"]),
    }

    # One-hot encode Location (baseline category maps to all zeros)
    location_col = f"Location_{inputs['location']}"
    row[location_col] = 1

    # One-hot encode Furnished (baseline category maps to all zeros)
    furnished_col = f"Furnished_{inputs['furnished']}"
    row[furnished_col] = 1

    df_row = pd.DataFrame([row])
    # Any column expected by the model but not set above defaults to 0
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    return df_row


# --------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------
model = load_model()
feature_columns = load_feature_columns()
categories = load_categories()

LOCATIONS = categories.get("Location", [])
FURNISHED_OPTIONS = categories.get("Furnished", [])

# --------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------
st.sidebar.header("Property Details")

area = st.sidebar.number_input(
    "Area (sqft)", min_value=200, max_value=10000, value=1500, step=50
)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
age = st.sidebar.slider("Age of property (years)", 0, 100, 10)
distance = st.sidebar.slider(
    "Distance to city center (km)", 0.0, 50.0, 5.0, step=0.5
)
location = st.sidebar.selectbox("Location", LOCATIONS)
furnished = st.sidebar.selectbox("Furnishing status", FURNISHED_OPTIONS)

st.sidebar.markdown("---")
st.sidebar.subheader("Amenities")
garage = st.sidebar.checkbox("Garage")  # kept for UI completeness (dropped in training)
garden = st.sidebar.checkbox("Garden")
pool = st.sidebar.checkbox("Swimming Pool")
security = st.sidebar.checkbox("Security System")
gym = st.sidebar.checkbox("Gym")

predict_clicked = st.sidebar.button("Predict Price", type="primary", use_container_width=True)

# --------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption(
    "Estimate a property's market price using a Linear Regression model "
    "trained on historical housing data."
)

with st.expander("How does this work?"):
    st.write(
        "This app loads a Linear Regression model trained on features like "
        "area, bedrooms, bathrooms, age, distance to city center, "
        "amenities, location and furnishing status. Fill in the property "
        "details in the sidebar and click **Predict Price**."
    )

st.divider()

if predict_clicked:
    inputs = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "distance": distance,
        "location": location,
        "furnished": furnished,
        "garden": garden,
        "pool": pool,
        "security": security,
        "gym": gym,
    }

    input_df = build_input_row(inputs, feature_columns)

    with st.spinner("Calculating estimate..."):
        prediction = float(model.predict(input_df)[0])
        prediction = max(prediction, 0)  # guard against negative predictions

    st.subheader("Estimated Price")
    col1, col2 = st.columns(2)
    col1.metric("Price (INR)", f"{prediction:,.0f}")
    col2.metric("Approx.", format_inr(prediction))

    with st.expander("View input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

else:
    st.info("Fill in the property details in the sidebar and click **Predict Price**.")

st.divider()
st.caption("Built with Streamlit · Model: scikit-learn Linear Regression")
