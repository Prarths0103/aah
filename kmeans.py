# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (Iris dataset as an example)
data = pd.read_csv('Iris.csv')
X = data.drop(columns=['Species'])  # Use the correct column names
y = data['Species']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------- K-Means Clustering -----------------
# Define the number of clusters (k)
k = 3  # You can change this to the number of clusters you want

# Initialize KMeans model
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the KMeans model
kmeans.fit(X_scaled)

# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Add the cluster labels to the dataset
X['Cluster'] = labels

# Print the cluster centers
print("Cluster Centers:")
print(cluster_centers)

# Visualizing the clusters (using the first two features for simplicity)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', label='Centroids')
plt.title("K-Means Clustering (2D Projection)")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Optionally, you can check the inertia (sum of squared distances of samples to their closest cluster center)
print("Inertia:", kmeans.inertia_)
