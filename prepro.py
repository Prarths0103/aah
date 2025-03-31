import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock

# Load dataset
data = pd.read_csv('Iris.csv')  # Replace with your actual dataset

data_numeric = data.select_dtypes(include=[np.number])  # Numeric columns only

data_cleaned = data.dropna()  # Remove rows with missing values

# Replace missing values with mean (for numeric columns)
data_filled = data.copy()
numeric_cols = data.select_dtypes(include=[np.number]).columns
data_filled[numeric_cols] = data_filled[numeric_cols].fillna(data_filled[numeric_cols].mean())

# Normalization
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data_filled[numeric_cols]), columns=numeric_cols)

# Correlation and Covariance
correlation_matrix = data_standardized.corr()
print("Corrlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ----------------- Covariance -----------------
# Calculate covariance matrix
covariance_matrix = data_standardized.cov()

# Display covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)

# Visualizing Covariance Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm')
plt.title("Covariance Matrix")
plt.show()

# Cosine Similarity
cosine_sim = cosine_similarity(data_standardized.iloc[:2, :])
print("Cosine Similarity:\n", cosine_sim)

# Proximal Analysis
print("Euclidean Distance:", euclidean(data_standardized.iloc[0], data_standardized.iloc[1]))
print("Manhattan Distance:", cityblock(data_standardized.iloc[0], data_standardized.iloc[1]))
print("Supremum:", np.max(data_numeric.to_numpy()))

# KNN Classification
X = data_standardized.iloc[:, :-1]
y = data_filled.iloc[:, -1]

if not pd.api.types.is_numeric_dtype(y):
    y = pd.factorize(y)[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find the best k using cross-validation
k_values = range(1, 21)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    scores.append(score)

best_k = k_values[np.argmax(scores)]  # Get k with the highest accuracy
print("Best k:", best_k)

# Train KNN model with best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

y_pred = knn_best.predict(X_test)
print("KNN Accuracy with best k:", knn_best.score(X_test, y_test))

# Plot accuracy vs k
plt.plot(k_values, scores, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Choosing the Best k")
plt.show()

#merge data
# Assuming you have another dataset 'other_data.csv'
other_data = pd.read_csv('ex.csv')

# Ensure there's a common column to merge on (e.g., 'ID')
if 'ID' in data_filled.columns and 'ID' in other_data.columns:
    merged_data = pd.merge(data_filled, other_data, on='ID', how='inner')
else:
    merged_data = pd.concat([data_filled, other_data], axis=1)  # Merge side by side

# Export the merged dataset to a CSV file
merged_data.to_csv('merged_data.csv', index=False)  # Replace with your desired file name

