import numpy as np
import matplotlib.pyplot as plt

def convergence_plot(mu_x, sigma_x, mu_y, sigma_y, n, xlabel='Height (cm)', ylabel='Weight (kg)'):
    np.random.seed(42)  # for reproducibility

    # Generate random samples
    X = np.random.normal(mu_x, sigma_x, n)
    Y = np.random.normal(mu_y, sigma_y, n)

    # Plot the samples
    plt.scatter(X, Y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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

# Example usage:
mu_x_input = float(input("Enter mu_x: "))
sigma_x_input = float(input("Enter sigma_x: "))
mu_y_input = float(input("Enter mu_y: "))
sigma_y_input = float(input("Enter sigma_y: "))
n_input = int(input("Enter n: "))
height_label_input = input("Enter the label for height (e.g., 'Height (cm)'): ")
weight_label_input = input("Enter the label for weight (e.g., 'Weight (kg)'): ")

convergence_plot(mu_x_input, sigma_x_input, mu_y_input, sigma_y_input, n_input, height_label_input, weight_label_input)