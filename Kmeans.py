import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def euclidean(point, data):

    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method
        self.centroids = [X_train[np.random.choice(range(len(X_train)))]]
        for _ in range(self.n_clusters-1):
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            dists /= np.sum(dists)
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

if __name__ == "__main__":
    
    from sklearn.datasets import load_iris

    # Load the Iris dataset
    iris = load_iris()
    input_data= iris.data  
    target_data= iris.target


    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=3)  
    kmeans.fit(input_data)
    centroids, cluster_labels = kmeans.evaluate(input_data)

    # Evaluate KMeans using silhouette score
    silhouette_avg = silhouette_score(input_data, cluster_labels)
    print("Silhouette Score:", silhouette_avg)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(input_data[:, 0],input_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], marker='o', c='red', s=200, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering on Iris Dataset')
    plt.show()
