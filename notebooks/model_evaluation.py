import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("data/cardio_processed.csv")

# Create X and y
X = data.drop("cardio", axis=1)
y = data["cardio"]

# Load trained model
with open("models/cardio_model.pkl", "rb") as file:
    model = pickle.load(file)

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
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# Print results
print("Model Evaluation Completed")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
print("\nResults saved in reports/model_results.txt")