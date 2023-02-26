import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import sklearn.metrics as metrics
from logistic_regression import logistic_regression

def load(filename):
    D = pd.read_csv(filename, sep = ",", header=0)
    y = D['Prediction'].to_numpy()
    iy = list(D.keys()).index("Prediction")

    D = D.to_numpy()
    x = np.hstack((D[:,1:iy], D[:,iy+1:]))

    return x,y

def Q2_5():
    x, y = load("emails.csv")

    test_x = x[4000:,:]
    test_y = y[4000:]

    train_x = x[:4000,:]
    train_y = y[:4000]

    model_knn = neighbors.KNeighborsClassifier(n_neighbors=5, )
    model_knn.fit(train_x, train_y)
    probs_knn = model_knn.predict_proba(test_x)
    preds_knn = probs_knn[:,1]
    fpr_knn, tpr_knn, _ = roc_curve(test_y, preds_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    model_lg = logistic_regression(train_x, train_y, 0.05)
    model_lg.train(5)
    preds_lg, _ = model_lg.predict(test_x)
    fpr_lg, tpr_lg, _ = roc_curve(test_y, preds_lg)
    roc_auc_lg = auc(fpr_lg, tpr_lg)

    plt.title('Q2-5 sol')
    plt.plot(fpr_knn, tpr_knn, color = 'blue', label = 'KNeighborsClassifier AUC = %0.2f' % roc_auc_knn)
    plt.plot(fpr_lg,  tpr_lg,  color = 'red',  label = 'LogisticRegression AUC = %0.2f' % roc_auc_lg)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (Positive label: 1)')
    plt.xlabel('False Positive Rate (Postive label: 1)')
    plt.show()

if __name__ == "__main__":
    Q2_5()