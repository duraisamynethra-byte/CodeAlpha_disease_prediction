from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", data.target_names)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed successfully!")
from sklearn.linear_model import LogisticRegression

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")
from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")
from sklearn.ensemble import RandomForestClassifier

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
from sklearn.svm import SVC

# Create and train the SVM model
svm_model = SVC()

svm_model.fit(X_train, y_train)

# Make predictions
svm_pred = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", round(svm_accuracy * 100, 2), "%")
print("\n--- Model Comparison ---")
print("Logistic Regression:", round(accuracy * 100, 2), "%")
print("Random Forest:", round(rf_accuracy * 100, 2), "%")
print("SVM:", round(svm_accuracy * 100, 2), "%")
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Logistic Regression Evaluation ---")

print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion matrix saved as confusion_matrix.png")
import joblib

# Save the trained model and scaler
joblib.dump(model, "disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as disease_model.pkl")
print("Scaler saved as scaler.pkl")