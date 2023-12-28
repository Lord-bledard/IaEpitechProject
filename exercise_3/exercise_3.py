import numpy as np

# Chargez les données à partir du fichier .npy
data = np.load('IaEpitechProject/exercise_3/data.npy')

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardiser les données
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Définir les méthodes de clustering
clustering_methods = [KMeans, AgglomerativeClustering]
method_names = ['KMeans', 'Agglomerative']

# Définir les heuristiques
heuristics = ['elbow', 'silhouette']

for method, method_name in zip(clustering_methods, method_names):
    for heuristic in heuristics:
        if heuristic == 'elbow' and method == KMeans:
            # Utiliser la méthode du Coude pour trouver le nombre optimal de clusters (seulement pour KMeans)
            inertias = []
            for n_clusters in range(2, 10):
                model = method(n_clusters=n_clusters)
                model.fit(data_standardized)
                inertias.append(model.inertia_)

            # Tracer la courbe du Coude
            plt.plot(range(2, 10), inertias, marker='o')
            plt.title(f'Méthode du Coude - {method_name}')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Inertie')
            plt.show()

        elif heuristic == 'silhouette':
            # Utiliser le score Silhouette pour trouver le nombre optimal de clusters
            silhouette_scores = []
            for n_clusters in range(2, 10):
                model = method(n_clusters=n_clusters)
                labels = model.fit_predict(data_standardized)
                silhouette_scores.append(silhouette_score(data_standardized, labels))

            # Tracer les scores Silhouette
            plt.plot(range(2, 10), silhouette_scores, marker='o')
            plt.title(f'Score Silhouette - {method_name}')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Score Silhouette')
            plt.show()

# Maintenant, vous pouvez utiliser la variable 'donnees' comme un tableau NumPy
