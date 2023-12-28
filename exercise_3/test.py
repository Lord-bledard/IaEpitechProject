import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les données
data = np.load('IaEpitechProject/exercise_3/data.npy')

# Normaliser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Méthode 1: K-Means
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Méthode 2: Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(data_scaled)

# Heuristique 1: Silhouette Score pour K-Means
silhouette_kmeans = silhouette_score(data_scaled, kmeans_labels)

# Heuristique 2: Calinski-Harabasz Index pour Hierarchical Clustering
calinski_agg = calinski_harabasz_score(data_scaled, agg_labels)

# Heuristique 3: Davies-Bouldin Index pour K-Means
davies_bouldin_kmeans = davies_bouldin_score(data_scaled, kmeans_labels)

# Heuristique 4: Adjusted Rand Index pour Hierarchical Clustering
# Remplacez true_labels par les vraies étiquettes si disponibles
X = np.array([0] * 400 + [1] * 400 + [2] * 400)
adjusted_rand_agg = adjusted_rand_score(X, agg_labels)

# Visualisation des résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolors='w')
plt.title(f'K-Means\nSilhouette Score: {silhouette_kmeans:.2f}\nDavies-Bouldin Index: {davies_bouldin_kmeans:.2f}')

plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=agg_labels, cmap='viridis', marker='o', edgecolors='w')
plt.title(f'Hierarchical Clustering\nCalinski-Harabasz Index: {calinski_agg:.2f}\nAdjusted Rand Index: {adjusted_rand_agg:.2f}')

plt.show()
