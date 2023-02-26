import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sklearn.metrics as metrics

def load(filename):
    D = pd.read_csv(filename, sep = ",", header=0)
    y = D['Prediction'].to_numpy()
    iy = list(D.keys()).index("Prediction")

    D = D.to_numpy()
    x = D[:,1:iy]
    x = np.hstack((x, D[:,iy+1:D.shape[0]]))

    return x,y

class logistic_regression():
    def __init__(self, x, y, lr):
        self.x = x
        self.x = self.normalize(self.x)
        self.y = y
        self.lr = lr
        self.theta = np.zeros(x.shape[1])

    def normalize(self, x):
        epsilon = 1e-100
        for i in range(3000):
            x[:,i] = (x[:,i] - x[:,i].mean(axis = 0)) / (x[:,i].std(axis = 0) + epsilon)
        return x

    def loss(self, f):
        loss = 0
        for i in range(f.shape[0]):
            if f[i] == 0:
                tmp = (1 - self.y[i]) * np.log(1 - f[i])
            elif f[i] == 1:
                tmp = self.y[i] * (np.log(f[i]))
            else:
                tmp = self.y[i] * np.log(f[i]) + (1 - self.y[i]) * np.log(1 - f[i])
            loss += tmp
        return -loss / f.shape[0]

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-(x.astype(float)) ))

    def calc_gradient(self):
        y_hat = self.sigmoid(self.x @ self.theta)
        gradient = self.x.T @ (- self.y + y_hat)
        gradient /= self.x.shape[0]
        return gradient, self.loss(y_hat)

    def train(self, times):
        for i in range(times):
            gl, l = self.calc_gradient()
            self.theta = np.subtract(self.theta, self.lr * gl)
            # _, predicted_y = self.predict(self.x)
            # print("theta:", self.theta)
            # print("loss:", l)
            # print("max gradient:", np.max(gl))
            # accuracy = accuracy_score(self.y, predicted_y)
            # print("accuracy:", accuracy)

    def predict(self, test_x):
        test_x = self.normalize(test_x)
        preds = self.sigmoid(test_x @ self.theta)
        predicted_y = [1 if x >= 0.5 else 0 for x in preds]
        return preds, predicted_y

def Q2_3():
    x,y = load("emails.csv")

    for i in range(5):
        s = max(0,i*1000-1)
        e = (i+1)*1000

        test_x = x[s:e,:]
        test_y = y[s:e]

        train_x = np.vstack((x[:s,:],x[e:,:]))
        train_y = np.hstack((y[:s],y[e:]))

        model = logistic_regression(train_x, train_y, 0.05)
        model.train(5)

        preds, predicted_y = model.predict(test_x)
        accuracy  = accuracy_score(test_y, predicted_y)
        precision = precision_score(test_y, predicted_y)
        recall    = recall_score(test_y, predicted_y)

        print("Fold",i+1,": accuracy-",accuracy," precision-",precision, " recall-",recall)
        # print(i+1,"&",accuracy,"&",precision,"&",recall," \\\\")
 
if __name__ == "__main__":
    Q2_3()
    