import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data (Make sure 'data' is defined above or read from csv)
data = pd.read_csv("data/cardio_processed.csv") 

# Create X and y
X = data.drop("cardio", axis=1)
y = data["cardio"]

# Load trained model safely using joblib
model = joblib.load("models/cardio_model.pkl")

# Predictions
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# Classification Report
report = classification_report(y, y_pred)

# Save results
with open("reports/model_results.txt", "w") as f:
    f.write(f"Accuracy Score: {accuracy}\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"Classification Report:\n{report}\n")

print("✅ Model evaluation completed and results saved!")