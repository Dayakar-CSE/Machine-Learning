import numpy as np
import matplotlib.pyplot as plt

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of error w.r.t predicted value (MSE)
def error_derivative(predicted, target):
    return 2 * (predicted - target)

# Derivative of sigmoid w.r.t input
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Update rule for weight
def update_weight(weight, gradient, learning_rate):
    return weight - learning_rate * gradient

# Inputs
x1, x2 = 0.1, 0.4
target = 0.7
learning_rate = 0.01

# Random initial weights
w1 = np.random.rand()
w2 = np.random.rand()

print("Initial Weights:", w1, w2)

# For plotting
predicted_output = []
network_error = []

# Training loop (gradient descent)
for _ in range(80000):
    # === Forward pass ===
    y = w1 * x1 + w2 * x2  # weighted sum
    predicted = sigmoid(y)  # activation
    error = (predicted - target) ** 2  # mean squared error

    # Save for plotting
    predicted_output.append(predicted)
    network_error.append(error)

    # === Backward pass ===
    dE_dP = error_derivative(predicted, target)
    dP_dY = sigmoid_derivative(y)

    # Gradients w.r.t weights
    grad_w1 = x1 * dP_dY * dE_dP
    grad_w2 = x2 * dP_dY * dE_dP

    # Weight update
    w1 = update_weight(w1, grad_w1, learning_rate)
    w2 = update_weight(w2, grad_w2, learning_rate)

# === Plotting ===
plt.figure()
plt.plot(network_error)
plt.title("Iteration vs Error")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(predicted_output)
plt.title("Iteration vs Prediction")
plt.xlabel("Iteration")
plt.ylabel("Prediction")
plt.grid(True)
plt.show()
