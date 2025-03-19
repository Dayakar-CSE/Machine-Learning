import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create a DataFrame with the provided data
data = {
    'VAR1': [1.713, 0.180, 0.353, 0.940, 1.486, 1.266, 1.540, 0.459, 0.773],
    'VAR2': [1.586, 1.786, 1.240, 1.566, 0.759, 1.106, 0.419, 1.799, 0.186],
    'CLASS': [0, 1, 1, 0, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Prepare the data for k-means (excluding the CLASS column)
X = df[['VAR1', 'VAR2']]

# Initialize k-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get the centroids
centroids = kmeans.cluster_centers_

# Predict the classification for the new point
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point)

# Find the class of the closest centroid based on the existing data
closest_centroid_index = predicted_cluster[0]
class_for_new_point = df[df['CLASS'] == closest_centroid_index].CLASS.mode()[0]

# Output results
print("Centroids:")
print(centroids)
print("Predicted Cluster Index:", closest_centroid_index)
print("Predicted Class for new point (VAR1=0.906, VAR2=0.606):", class_for_new_point)

# Optional: Visualize the clusters
plt.scatter(df['VAR1'], df['VAR2'], c=df['CLASS'], label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='Centroids', marker='X', s=200)
plt.scatter(new_point[0][0], new_point[0][1], c='green', label='New Point', marker='o', s=100)
plt.title("K-Means Clustering")
plt.xlabel("VAR1")
plt.ylabel("VAR2")
plt.legend()
plt.show()
