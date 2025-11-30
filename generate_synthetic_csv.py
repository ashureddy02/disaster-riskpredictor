import pandas as pd
import random

rows = 1000
data = []

weather_options = [
    "Sunny", "Cloudy", "Rainy", "Windy", "Foggy",
    "Thunderstorm", "Cyclone", "Heat Wave", "Cold Wave", "Extreme Rainfall", "Stormy"
]

for i in range(rows):
    temperature = round(random.uniform(15, 45), 1)
    weather = random.choice(weather_options)
    rainfall = round(random.uniform(0, 500), 1)
    river_level = round(random.uniform(0, 15), 2)
    wind = round(random.uniform(0, 150), 1)
    soil_moist = round(random.uniform(0, 100), 1)
    past_floods = random.randint(0, 5)

    if rainfall > 400 or river_level > 12 or wind > 120 or weather == "Stormy":
        risk = "Severe"
    elif rainfall > 250 or river_level > 8 or wind > 80:
        risk = "High"
    elif rainfall > 120 or river_level > 5 or soil_moist > 60 or past_floods >= 2 or weather == "Rainy":
        risk = "Medium"
    else:
        risk = "Low"

    data.append([temperature, weather, rainfall, river_level, wind, soil_moist, past_floods, risk])

df = pd.DataFrame(data, columns=[
    "temperature_deg", "weather", "rainfall_mm", "river_level_m",
    "wind_kmph", "soil_moisture", "past_floods", "risk_level"
])

df.to_csv("synthetic_disaster_risk.csv", index=False)
print("✅ synthetic_disaster_risk.csv created successfully!")
