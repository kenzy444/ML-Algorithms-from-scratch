import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X, y = datasets.make_classification(n_samples=1000,  n_features=10, n_classes=2,random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )
    knn= KNN()
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    print("Accuracy score is :", accuracy_score(y_test, predictions))
    print("Recall score is :", recall_score(y_test, predictions))
    print("Precision score is :", precision_score(y_test, predictions))

