import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

print("LIME Analysis Started")

df = pd.read_csv("data/cardio_processed.csv")

X = df.drop("cardio", axis=1)
y = df["cardio"]

model = joblib.load("models/cardio_model.pkl")

explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=["No Disease", "Disease"],
    mode="classification"
)

exp = explainer.explain_instance(
    X.iloc[0].values,
    model.predict_proba
)

exp.save_to_file("reports/lime_explanation.html")

print("LIME explanation saved!")