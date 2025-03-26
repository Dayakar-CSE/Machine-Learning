import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create a DataFrame with the provided data
data = {
    'VAR1': [1.713, 0.180, 0.353, 0.940, 1.486, 1.266, 1.540, 0.459, 0.773],
    'VAR2': [1.586, 1.786, 1.240, 1.566, 0.759, 1.106, 0.419, 1.799, 0.186],
    'CLASS': [0, 1, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Prepare data for K-Means clustering (excluding the CLASS column)
X = df[['VAR1', 'VAR2']]

# Initialize K-Means with 3 clusters and fit the model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Predict the cluster for a new data point
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point)[0]

# Determine the class of the closest centroid based on majority class in that cluster
df['Cluster'] = kmeans.labels_
cluster_class_mode = df[df['Cluster'] == predicted_cluster]['CLASS'].mode()[0]

# Output results
print("Cluster Centroids:\n", centroids)
print("Predicted Cluster Index:", predicted_cluster)
print("Predicted Class for new point (VAR1=0.906, VAR2=0.606):", cluster_class_mode)

# Visualization
plt.scatter(df['VAR1'], df['VAR2'], c=df['Cluster'], cmap='viridis', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.scatter(new_point[0, 0], new_point[0, 1], c='green', marker='o', s=100, label='New Point')
plt.title("K-Means Clustering")
plt.xlabel("VAR1")
plt.ylabel("VAR2")
plt.legend()
plt.show()
