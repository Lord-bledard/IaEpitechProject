import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

# Parameters
mu_x, sigma_x = 170, 10
mu_y, sigma_y = 70, 5
n = 1000

# Generate random samples
X = np.random.normal(mu_x, sigma_x, n)
Y = np.random.normal(mu_y, sigma_y, n)

# Plot the samples
plt.scatter(X, Y, alpha=0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Random Samples from Z = (X, Y)')
plt.show()

# Compute empirical averages and distances
empirical_averages = np.cumsum(np.vstack((X, Y)), axis=1) / np.arange(1, n + 1)
expected_value = np.array([mu_x, mu_y])

# Compute distances for each sample size
distances = np.linalg.norm(empirical_averages - expected_value[:, np.newaxis], axis=0)

# Plot the convergence
plt.plot(np.arange(1, n + 1), distances)
plt.xlabel('Number of Samples (n)')
plt.ylabel('Euclidean Distance')
plt.title('Convergence of Empirical Average to Expected Value')
plt.show()