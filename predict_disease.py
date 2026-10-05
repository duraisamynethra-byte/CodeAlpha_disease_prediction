import joblib
from sklearn.datasets import load_breast_cancer

# Load the trained model and scaler
model = joblib.load("disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset
data = load_breast_cancer()

print("Disease Prediction System")
print("-------------------------")

# Use one sample patient from the dataset
sample = data.data[0]

# Scale the sample
sample_scaled = scaler.transform([sample])

# Make prediction
prediction = model.predict(sample_scaled)
probability = model.predict_proba(sample_scaled)

# Display result
if prediction[0] == 0:
    result = "Malignant"
else:
    result = "Benign"

confidence = max(probability[0]) * 100

print("Predicted Result:", result)
print("Prediction Confidence:", round(confidence, 2), "%")