import pandas as pd
import joblib
import shap

print("SHAP Analysis Started")

df = pd.read_csv("data/cardio_processed.csv")

X = df.drop(columns=["cardio"])

model = joblib.load("models/cardio_model.pkl")

explainer = shap.Explainer(model, X)

shap_values = explainer(X)

print("SHAP values generated successfully!")