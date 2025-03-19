import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
data = iris.data
labels = iris.target

for i in [0, 79, 99, 101]:
    print(f"index: {i:3}, features: {data[i]}, label: {labels[i]}")

np.random.seed(42)
indices = np.random.permutation(len(data))
n_training_samples = 12
learn_data = data[indices[:-n_training_samples]]
learn_labels = labels[indices[:-n_training_samples]]
test_data = data[indices[-n_training_samples:]]
test_labels = labels[indices[-n_training_samples:]]

print("The first samples of our learn set:")
print(f"{'index':7s}{'data':20s}{'label':3s}")
for i in range(5):
    print(f"{i:4d} {learn_data[i]} {learn_labels[i]:3}")

print("The first samples of our test set:")
print(f"{'index':7s}{'data':20s}{'label':3s}")
for i in range(5):
    print(f"{i:4d} {test_data[i]} {test_labels[i]:3}")

# Visualize the data
colours = ("r", "g", "y")
X = []
for iclass in range(3):
    X.append([[], [], []])
for i in range(len(learn_data)):
    if learn_labels[i] == iclass:
        X[iclass][0].append(learn_data[i][0])
        X[iclass][1].append(learn_data[i][1])
        X[iclass][2].append(sum(learn_data[i][2:]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()

def distance(instance1, instance2):
    """ Calculates the Euclidean distance between two instances"""
    return np.linalg.norm(np.subtract(instance1, instance2))

def get_neighbors(training_set, labels, test_instance, k, distance):
    """get_neighbors calculates a list of the k nearest neighbors of an instance 'test_instance'."""
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

for i in range(5):
    neighbors = get_neighbors(learn_data, learn_labels, test_data[i], 3, distance=distance)
    print("Index: ", i, '\n', "Testset Data: ", test_data[i], '\n', "Testset Label: ", test_labels[i], '\n', "Neighbors: ", neighbors, '\n')
