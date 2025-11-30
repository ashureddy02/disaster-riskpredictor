import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1) Load data
df = pd.read_csv("synthetic_disaster_risk.csv")

# 2) Split X and y
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# 3) Encode weather (input feature)
weather_encoder = LabelEncoder()
X["weather"] = weather_encoder.fit_transform(X["weather"])

# 4) Encode target (risk_level)
risk_encoder = LabelEncoder()
y_encoded = risk_encoder.fit_transform(y)

# 5) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 6) Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7) Evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClasses (encoded):", list(risk_encoder.classes_))
print("\nClassification report:\n")
labels_used = sorted(set(y_test))  # only classes present in test
print(classification_report(
    y_test, preds, labels=labels_used, target_names=[risk_encoder.classes_[i] for i in labels_used]
))


import joblib

joblib.dump(model, "disaster_risk_model.pkl")
joblib.dump(weather_encoder, "weather_encoder.pkl")
joblib.dump(risk_encoder, "risk_encoder.pkl")



