import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load data and labels
data = np.load('data.npy')
labels = np.load('labels.npy')

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Dimensionality reduction to 2 dimensions using PCA
pca_2d = PCA(n_components=2)
pca_result_2d = pca_2d.fit_transform(data_standardized)

# Dimensionality reduction to 3 dimensions using PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(data_standardized)

# Dimensionality reduction to 2 dimensions using t-SNE
tsne_2d = TSNE(n_components=2, random_state=42)
tsne_result_2d = tsne_2d.fit_transform(data_standardized)

# Dimensionality reduction to 3 dimensions using t-SNE
tsne_3d = TSNE(n_components=3, random_state=42)
tsne_result_3d = tsne_3d.fit_transform(data_standardized)

# Plotting
plt.figure(figsize=(12, 6))

# PCA 2D
plt.subplot(2, 3, 1)
plt.scatter(pca_result_2d[:, 0], pca_result_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('PCA 2D')

# PCA 3D
ax = plt.subplot(2, 3, 2, projection='3d')
ax.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], pca_result_3d[:, 2], c=labels, cmap='viridis', alpha=0.5)
ax.set_title('PCA 3D')

# t-SNE 2D
plt.subplot(2, 3, 4)
plt.scatter(tsne_result_2d[:, 0], tsne_result_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('t-SNE 2D')

# t-SNE 3D
ax = plt.subplot(2, 3, 5, projection='3d')
ax.scatter(tsne_result_3d[:, 0], tsne_result_3d[:, 1], tsne_result_3d[:, 2], c=labels, cmap='viridis', alpha=0.5)
ax.set_title('t-SNE 3D')

plt.tight_layout()
plt.show()