import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Minimal styling
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Paths
# -------------------------
MODEL_PATH = "disaster_risk_model.pkl"
WEATHER_ENCODER_PATH = "weather_encoder.pkl"
RISK_ENCODER_PATH = "risk_encoder.pkl"

# -------------------------
# Load model & encoders (safe)
# -------------------------
model = None
weather_encoder = None
risk_encoder = None
model_loaded = False
enc_loaded = False
risk_enc_loaded = False

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_loaded = True
except Exception as e:
    st.warning(f"Could not load model from {MODEL_PATH}: {e}")

try:
    if os.path.exists(WEATHER_ENCODER_PATH):
        weather_encoder = joblib.load(WEATHER_ENCODER_PATH)
        enc_loaded = True
except Exception as e:
    st.warning(f"Could not load weather encoder from {WEATHER_ENCODER_PATH}: {e}")

try:
    if os.path.exists(RISK_ENCODER_PATH):
        risk_encoder = joblib.load(RISK_ENCODER_PATH)
        risk_enc_loaded = True
except Exception as e:
    st.warning(f"Could not load risk encoder from {RISK_ENCODER_PATH}: {e}")

# Fallback weather classes if encoder missing
if enc_loaded and hasattr(weather_encoder, "classes_"):
    WEATHER_CLASSES = list(weather_encoder.classes_)
else:
    WEATHER_CLASSES = ["sunny", "partly_cloudy", "rainy", "stormy", "foggy", "windy", "snow"]

FRIENDLY_ICON_MAP = {
    "sunny": "☀️ Sunny",
    "partly_cloudy": "🌤️ Partly Cloudy",
    "rainy": "🌧️ Rainy",
    "stormy": "⛈️ Stormy",
    "foggy": "🌫️ Foggy",
    "windy": "🌪️ Windy/Strong winds",
    "snow": "❄️ Snow/Ice",
}

WEATHER_DISPLAY = []
for cls in WEATHER_CLASSES:
    display = FRIENDLY_ICON_MAP.get(cls, cls)
    WEATHER_DISPLAY.append((display, cls))

# -------------------------
# Helpers
# -------------------------
def color_for_label(label: str) -> str:
    mapping = {
        "Low": "#2ecc71",
        "Medium": "#f1c40f",
        "High": "#e67e22",
        "Severe": "#e74c3c"
    }
    return mapping.get(label, "#95a5a6")

def build_input_df(temperature, weather, rainfall, river_level, wind, soil_moist, past_floods):
    df = pd.DataFrame(
        [[temperature, weather, rainfall, river_level, wind, soil_moist, past_floods]],
        columns=[
            "temperature_deg",
            "weather",
            "rainfall_mm",
            "river_level_m",
            "wind_kmph",
            "soil_moisture",
            "past_floods",
        ],
    )
    return df

def safe_transform_weather(series):
    """Use the encoder if available, otherwise return series as-is."""
    if enc_loaded and weather_encoder is not None:
        try:
            return weather_encoder.transform(series)
        except Exception:
            try:
                return weather_encoder.transform(series.values.reshape(-1, 1))
            except Exception:
                st.info("Weather encoder transform failed; using raw weather values.")
                return series
    else:
        return series

def interpret_with_feature_importances(model, input_df):
    items = []
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.array(model.feature_importances_)
            names = list(input_df.columns)
            idx_sorted = np.argsort(fi)[::-1][:5]
            for idx in idx_sorted:
                name = names[idx] if idx < len(names) else f"feature_{idx}"
                items.append((name, float(fi[idx])))
    except Exception:
        pass
    return items

def heuristic_label(input_df):
    """Simple rules fallback for label when model missing or fails."""
    row = input_df.iloc[0]
    temp = float(row["temperature_deg"])
    rain = float(row["rainfall_mm"])
    wind = float(row["wind_kmph"])
    river = float(row["river_level_m"])
    soil = float(row["soil_moisture"])
    past = int(row["past_floods"])
    score = 0
    if rain > 50: score += 3
    elif rain > 10: score += 1
    if river > 3: score += 3
    if soil > 70: score += 1
    if wind > 80: score += 2
    if past >= 1: score += 2
    if score >= 6:
        return "Severe"
    elif score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"

def map_pred_to_label(pred_val):
    """
    Map whatever the model/encoder returns to one of: Low, Medium, High, Severe.
    pred_val may be numeric (0/1/2/3), string, or already a label.
    """
    # If risk_encoder exists, try to use it
    if risk_enc_loaded and risk_encoder is not None:
        try:
            # Try to inverse transform directly (works when pred_val is encoded int)
            inv = risk_encoder.inverse_transform([pred_val])
            return str(inv[0])
        except Exception:
            # Maybe pred_val is numeric in string form, try convert
            try:
                inv = risk_encoder.inverse_transform([int(pred_val)])
                return str(inv[0])
            except Exception:
                pass

    # If no encoder or previous failed, try heuristics:
    # If it's already a label like "Low"/"low"/"LOW", normalize to Title case.
    if isinstance(pred_val, str):
        cleaned = pred_val.strip().lower()
        if cleaned in {"low", "medium", "high", "severe"}:
            return cleaned.title()

    # If it's numeric (numpy int etc.), map using sensible default
    try:
        num = int(pred_val)
        mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}
        return mapping.get(num, "Unknown")
    except Exception:
        pass

    # Fallback
    return "Unknown"

# -------------------------
# Page setup + UI
# -------------------------
st.set_page_config(page_title="Disaster Risk Predictor", layout="centered")
st.title("🌊 Disaster Risk Predictor")

st.write(
    "Enter the **current weather and environment details** for a location.\n\n"
    "The model will classify the **disaster risk level** as: `Low`, `Medium`, `High`, or `Severe`."
)

st.info(
    "➡️ **How to use:**\n"
    "1. Fill in the fields below (type or use + / –).  \n"
    "2. Choose the current weather from the dropdown.  \n"
    "3. Click **Predict Disaster Risk** to see the result (color-coded)."
)

st.divider()

# Input controls
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("🌡 Temperature (°C)", min_value=-20.0, max_value=60.0, value=28.0, step=0.5)
    rainfall = st.number_input("🌧 Rainfall (mm) — recent / last 3 hours", min_value=0.0, max_value=200.0, value=5.0, step=0.1)
    wind = st.number_input("💨 Wind Speed (kmph)", min_value=0.0, max_value=150.0, value=10.0, step=0.5)

with col2:
    river_level = st.number_input("🏞 River Level (m)", min_value=0.0, max_value=15.0, value=0.5, step=0.1)
    soil_moist = st.number_input("🌱 Soil Moisture (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
    past_floods = st.slider("🌊 Past Flood Count (in last few years)", min_value=0, max_value=8, value=0)

display_names = [d for d, v in WEATHER_DISPLAY]
value_map = {d: v for d, v in WEATHER_DISPLAY}
selected_display = st.selectbox("⛅ Current Weather Condition", display_names, index=0)
weather_value = value_map[selected_display]

st.divider()

if model_loaded:
    st.success("Model loaded. Predictions will use your saved model.")
else:
    st.info(
        "No trained model found — the app will use a heuristic fallback so the UI still demonstrates behaviour. "
        f"To enable model predictions, add the trained model file at: {MODEL_PATH}"
    )

# Predict
if st.button("🔍 Predict Disaster Risk", use_container_width=True):
    input_df = build_input_df(
        temperature, weather_value, rainfall, river_level, wind, soil_moist, past_floods
    )

    # Try to encode weather if encoder present
    try:
        if enc_loaded and weather_encoder is not None:
            input_df["weather"] = safe_transform_weather(input_df["weather"])
    except Exception as e:
        st.info(f"Warning: weather encoding failed: {e}")

    predicted_label = None
    predicted_proba = None
    contributions = []

    if model_loaded and model is not None:
        try:
            # If model can give probabilities, use them
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                idx = np.argmax(proba, axis=1)[0]  # index in classes_
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    # classes[idx] may be numeric or encoded; map to text label
                    raw_pred = classes[idx]
                    predicted_label = map_pred_to_label(raw_pred)
                else:
                    # fallback to using idx as numeric label
                    predicted_label = map_pred_to_label(idx)

                predicted_proba = float(np.max(proba))
            else:
                preds = model.predict(input_df)
                raw_pred = preds[0]
                predicted_label = map_pred_to_label(raw_pred)
        except Exception as e:
            st.info(f"Model prediction failed, falling back to heuristic: {e}")
            predicted_label = None

        contributions = interpret_with_feature_importances(model, input_df)
    else:
        predicted_label = None

    if predicted_label is None or predicted_label == "Unknown":
        predicted_label = heuristic_label(input_df)

    # Ensure predicted_label is one of the expected four (title-cased)
    if predicted_label.strip().lower() not in {"low", "medium", "high", "severe"}:
        # enforce heuristic if mapping still didn't resolve
        predicted_label = heuristic_label(input_df)

    color = color_for_label(predicted_label)
    st.markdown(
        f"""
        <div style="border-radius:10px;padding:16px;background:{color};color:#ffffff">
            <h2 style="margin:0">⚠️ Predicted Risk: {predicted_label}</h2>
            {"<p style='margin:6px 0;font-weight:600'>Confidence: {:.1%}</p>".format(predicted_proba) if predicted_proba is not None else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predicted_label in ["High", "Severe"]:
        st.warning(
            "- Take precautions. Monitor weather warnings.\n"
            "- Consider moving to higher grounds."
        )
    elif predicted_label == "Medium":
        st.info(
            "- Conditions are somewhat risky.\n"
            "- Monitor updates."
        )
    else:
        st.success("- Low immediate risk. Continue monitoring.")

    st.markdown("**Inputs summary**")
    summary_df = pd.DataFrame({
        "feature": ["temperature (°C)", "weather", "rainfall (mm)", "river_level (m)",
                    "wind (kmph)", "soil_moisture (%)", "past_floods"],
        "value": [temperature, selected_display, rainfall, river_level,
                  wind, soil_moist, past_floods]
    })
    st.table(summary_df)

    if len(contributions) > 0:
        st.markdown("**Top contributing factors (explainability)**")
        for name, imp in contributions:
            st.markdown(f"- **{name}** (importance ≈ {imp:.2f})")
    else:
        st.write("No feature importances available from model.")

    st.caption("Prediction generated using your inputs above.")
