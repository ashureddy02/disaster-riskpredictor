# 🌊 Disaster Risk Predictor (Machine Learning + Streamlit)

A machine-learning powered web app that predicts **disaster risk levels** based on real-time environmental and weather inputs.

The model classifies risk into four levels:

- **Low**
- **Medium**
- **High**
- **Severe**

This project uses a **Random Forest Classifier** trained on a **synthetic disaster dataset**, and provides a clean Streamlit interface for real-time prediction.

---

## 📊 Synthetic Dataset

This project includes a custom-generated dataset:

📄 `synthetic_disaster_risk.csv`

This dataset is **not collected from real-world sources**.  
Instead, it is created using:

📌 `generate_synthetic_csv.py`

The script simulates realistic weather & environmental conditions such as:

- Temperature  
- Rainfall  
- Soil moisture  
- River levels  
- Wind speed  
- Past flood count  
- Weather category (sunny, rainy, stormy, etc.)

This dataset is used to train:

- `disaster_risk_model.pkl`
- `weather_encoder.pkl`
- `risk_encoder.pkl`

> **Why synthetic?**  
> Synthetic data is useful when real datasets are unavailable or for rapid prototyping ML models.

---

## 🚀 Features

✔ Predicts risk level (`Low`, `Medium`, `High`, `Severe`)  
✔ Uses machine learning + fallback heuristic  
✔ Color-coded results (green/yellow/orange/red)  
✔ Displays model confidence (if available)  
✔ Friendly UI with emojis  
✔ Includes feature importance analysis  
✔ Works locally or can be deployed to Streamlit Cloud  

---

## 📁 Project Structure

DisasterRiskPredictorML/
│
├── app.py # Streamlit web app
├── train_model.py # Train Random Forest model
├── predict.py # Local prediction helper
├── generate_synthetic_csv.py # Generates synthetic CSV dataset
├── synthetic_disaster_risk.csv # Training dataset (synthetic)
├── disaster_risk_model.pkl # Trained model
├── weather_encoder.pkl # Encoder for weather column
├── risk_encoder.pkl # Encoder for risk labels
└── README.md

yaml
Copy code

---
Future improvements:
## 🐳 Docker (Coming Soon)

This project will soon include full Docker support so it can be containerized and run anywhere without installing Python or dependencies manually.

Dockerizing the app will include:

- A `Dockerfile` to build the image  
- Installing Python + dependencies inside the container  
- Exposing the Streamlit server port  
- Running the app with `streamlit run app.py` inside the container  
- Commands like:

```bash
docker build -t disaster-risk .
docker run -p 8501:8501 disaster-risk
--

## 🧠 Machine Learning Model

This project uses a **Random Forest Classifier** trained on a custom synthetic dataset.  
The model learns patterns from the following features:

- 🌡 **Temperature (°C)**
- ⛅ **Weather condition**
- 🌧 **Rainfall (mm)**
- 🏞 **River level (m)**
- 💨 **Wind speed (km/h)**
- 🌱 **Soil moisture (%)**
- 🌊 **Past flood count**

### 🎯 Prediction Output  
The model classifies the overall disaster risk into one of the four categories:

- **Low**
- **Medium**
- **High**
- **Severe**

These labels are encoded during training and decoded during prediction for user-friendly visual output.



