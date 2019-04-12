import numpy as np;
class RidgeRegression:
    def __init__(self) :
	    self.Weight = []
    def fitting(self, X, y, lamb):
        X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1) 
        y = np.array(y).reshape(len(y), 1)
        self.Weight= np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(len(X[0]))), X.T), y)  
    def predict(self, X) :

	    X = np.append(np.ones((len(X), 1)), np.array(X), axis = 1)

	    return np.dot(X, self.Weight)
    
    def Error(self, X, y):
        y_hat = (self.predict(X).reshape(len(X)) > 0) * 2 - 1
        y = np.array(y)
        return 1 - np.sum(np.abs((y_hat + y) / 2)) / float(len(X))
        