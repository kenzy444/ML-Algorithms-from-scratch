import numpy as np 


class LinearRegression:
    def __init__(self, learning_rate= 0.01, n_iters = 1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bais = None

    def fit(self, X, y):
        # we have a weight for each feature
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bais = 0
        # y = wX + b
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bais 
            # clacultae the error and update the weights with gradient descente 
            dw = (1/n_samples) * np.dot(X.T, (y_pred- y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bais = self.bais - self.learning_rate * db 

    def predict(self, X):
            y_pred = np.dot(X, self.weights) + self.bais 
            return y_pred


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    

    X, y = datasets.make_regression(n_samples = 100, n_features=1,noise =20, random_state= 4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    LR =LinearRegression(learning_rate= 0.015)
    LR.fit(X_train, y_train) 
    predictions = LR.predict(X_test)
    
    def mse(y_test , predictions):
         return np.mean(y_test - predictions)**2
    

    mse = mse(y_test, predictions)
    print(mse)
    import matplotlib.pyplot as plt 
    y_pred_line = LR.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()