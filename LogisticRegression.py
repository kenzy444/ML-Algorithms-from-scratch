import numpy as np 


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None 
        self.bias =0.0 
        

    def fit(self, X ,y ):
        n_samples , n_features = X.shape
        self.bais = 0.0
        self.weights = np.zeros(n_features)
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights)+ self.bias
            # apply segmoid function  
            y_pred = 1 / (1+ np.exp(-linear_pred))
            # calculate the gradients 
            dw = (1 / n_samples) * np.dot(X.T ,(y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # updates 
            self.weights -= self.learning_rate  *dw
            self.bias -= self.learning_rate *db 
            

    def predict(self, X):
          linear_pred = np.dot(X, self.weights)+ self.bias 
          y_pred = 1 / (1+ np.exp(-linear_pred)) #what we have in predictions is the probability so we have the treshold 0.5 if > is 1 else 0
          predictions = [0 if y <=0.5 else 1 for y in y_pred] 
          return predictions


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = datasets.load_breast_cancer()
    X , y = data.data , data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    LR = LogisticRegression()
    LR.fit(X_train, y_train) 
    predictions =  LR.predict(X_test)
    print("accuracy is :" ,accuracy_score(predictions, y_test) )
