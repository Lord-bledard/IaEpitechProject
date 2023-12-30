import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load('exercise_3/data.npy')

# Normaliser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Réduction de dimension
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Méthode 1: K-Means
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Méthode 2: Agglomerative Hierarchical Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(data_scaled)

# Silhouette Score pour K-Means
silhouette_kmeans = silhouette_score(data_scaled, kmeans_labels)

# Calinski-Harabasz Index pour Hierarchical Clustering
calinski_agg = calinski_harabasz_score(data_scaled, agg_labels)

# Davies-Bouldin Index pour K-Means
davies_bouldin_kmeans = davies_bouldin_score(data_scaled, kmeans_labels)

# Adjusted Rand Index pour Hierarchical Clustering
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

'''
# Évaluation de la stabilité des clusters pour K-Means
n_init_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
silhouette_scores = []

for n_init in n_init_values:
    kmeans_stable = KMeans(n_clusters=3, n_init=n_init, random_state=42)
    labels_stable = kmeans_stable.fit_predict(data_scaled)
    silhouette_scores.append(silhouette_score(data_scaled, labels_stable))

# Visualisation des scores de silhouette 
plt.plot(n_init_values, silhouette_scores, marker='o')
plt.xlabel('Nombre d\'initialisations')
plt.ylabel('Silhouette Score')
plt.title('Stabilité des clusters avec différentes initialisations pour K-Means')
plt.show()
'''