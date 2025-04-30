# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate random dataset
np.random.seed(0)  # For reproducibility
x = np.random.rand(100, 1)                          # 100 random numbers between 0 and 1
y = 2 + 3 * x + np.random.rand(100, 1)              # Linear relation with some noise

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Make predictions
y_pred = model.predict(x)

# Evaluate the model
rmse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print model parameters and evaluation metrics
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score (R²):", r2)

# Plot the results
plt.scatter(x, y, color='blue', label='Actual data', s=10)
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('x values (0 to 1)')
plt.ylabel('y values (approx. 2 to 5)')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()
