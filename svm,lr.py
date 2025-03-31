import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (replace with your actual dataset)
# Here we use a simple example dataset
data = pd.read_csv('ex.csv')  # Replace with your dataset file path

# Print first few rows of the data to inspect
print(data.head())

# Assuming you have some features (X) and target variable (y)
# Replace 'target_column' with your actual target column name
X = data.drop(columns=['Age'])  # Features (excluding the target column)
y = data['Age']  # Target variable

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Support Vector Machine (SVM) --------------------

# Initialize and train SVM model (for classification)
svm = SVC(kernel='linear')  # You can change the kernel to 'rbf' or 'poly' for different behavior
svm.fit(X_train, y_train)

# Predict on the test data
y_pred_svm = svm.predict(X_test)

# Confusion Matrix and Accuracy for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Plot confusion matrix for SVM
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -------------------- Linear Regression --------------------

# Initialize and train Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Predict on the test data
y_pred_lr = linear_reg.predict(X_test)

# Calculate the accuracy (for regression, we typically use R^2 score instead of accuracy)
r2 = r2_score(y_test, y_pred_lr)
print("Linear Regression R^2 Score:", r2)

# -------------------- Linear Regression Graph --------------------

# Plotting the true values vs predicted values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, color='blue')  # Plot actual vs predicted values
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Line of perfect prediction
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
