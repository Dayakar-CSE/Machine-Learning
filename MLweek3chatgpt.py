import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset
iris = datasets.load_iris()
data, labels = iris.data, iris.target

# Display sample data points
for i in [0, 79, 99, 101]:
    print(f"Index: {i:3}, Features: {data[i]}, Label: {labels[i]}")

# Shuffle data randomly
np.random.seed(42)
indices = np.random.permutation(len(data))
n_training_samples = 12

# Split into training and testing datasets
learn_data = data[indices[:-n_training_samples]]
learn_labels = labels[indices[:-n_training_samples]]
test_data = data[indices[-n_training_samples:]]
test_labels = labels[indices[-n_training_samples:]]

# Display first few training samples
print("Training Set Samples:")
for i in range(5):
    print(f"{i:4d} {learn_data[i]} {learn_labels[i]:3}")

# Display first few test samples
print("Test Set Samples:")
for i in range(5):
    print(f"{i:4d} {test_data[i]} {test_labels[i]:3}")

# Visualizing training data in 3D
colours = ("r", "g", "y")
X = [[[], [], []] for _ in range(3)]

for i in range(len(learn_data)):
    X[learn_labels[i]][0].append(learn_data[i][0])
    X[learn_labels[i]][1].append(learn_data[i][1])
    X[learn_labels[i]][2].append(sum(learn_data[i][2:]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()

# Function to calculate Euclidean distance
def euclidean_distance(instance1, instance2):
    return np.linalg.norm(np.subtract(instance1, instance2))

# Function to find k nearest neighbors
def get_neighbors(training_set, labels, test_instance, k):
    distances = [(training_set[i], euclidean_distance(test_instance, training_set[i]), labels[i]) 
                 for i in range(len(training_set))]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    return distances[:k]

# Testing the KNN algorithm
for i in range(5):
    neighbors = get_neighbors(learn_data, learn_labels, test_data[i], 3)
    print(f"Index: {i}\nTest Data: {test_data[i]}\nTest Label: {test_labels[i]}\nNeighbors: {neighbors}\n")
