import joblib
import pandas as pd

# 1️⃣ Load saved model + encoders
model = joblib.load("disaster_risk_model.pkl")
weather_encoder = joblib.load("weather_encoder.pkl")
risk_encoder = joblib.load("risk_encoder.pkl")

# 2️⃣ ----- CHANGE THESE VALUES TO TEST DIFFERENT CASES -----
temperature_deg = 30.5
weather = "Rainy"          # try: Sunny / Cloudy / Stormy / Windy etc.
rainfall_mm = 320.0
river_level_m = 9.2
wind_kmph = 85.0
soil_moisture = 70.0
past_floods = 2
# -----------------------------------------------------------

# 3️⃣ Put values into a DataFrame with same columns as training
input_df = pd.DataFrame([{
    "temperature_deg": temperature_deg,
    "weather": weather,
    "rainfall_mm": rainfall_mm,
    "river_level_m": river_level_m,
    "wind_kmph": wind_kmph,
    "soil_moisture": soil_moisture,
    "past_floods": past_floods
}])

# 4️⃣ Encode weather using the same encoder
input_df["weather"] = weather_encoder.transform(input_df["weather"])

# 5️⃣ Predict
pred_encoded = model.predict(input_df)[0]
pred_label = risk_encoder.inverse_transform([pred_encoded])[0]

print("Input conditions:")
print(input_df)
print("\n👉 Predicted Risk Level:", pred_label)
