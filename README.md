# MACHINE-LEARNING-MODEL-IMPLEMENTATION
# Task 4 – Machine Learning Model Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("diabetes.csv")
print("Dataset Head:\n", data.head())
X = data.drop("Outcome", axis=1)   # features
y = data["Outcome"]                # target (0 = No Diabetes, 1 = Diabetes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
acc_log = accuracy_score(y_test, y_pred_log)
acc_dt = accuracy_score(y_test, y_pred_dt)
plt.bar(["Logistic Regression", "Decision Tree"], [acc_log, acc_dt], color=['blue','green'])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()
plt.figure(figsize=(12,8))
plot_tree(dt, filled=True, feature_names=data.columns[:-1], class_names=["No Diabetes", "Diabetes"])
plt.show()
sample = np.array([[2, 120, 70, 20, 80, 30.5, 0.5, 35]])  
sample = scaler.transform(sample)
print("\nSample Prediction:")
print("Logistic Regression:", "Diabetic" if log_reg.predict(sample)[0] == 1 else "Not Diabetic")
print("Decision Tree:", "Diabetic" if dt.predict(sample)[0] == 1 else "Not Diabetic")
