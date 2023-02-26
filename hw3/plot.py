import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric

def load(filename):
    D = pd.read_csv(filename, sep = " ", header = None)
    D = D.to_numpy()
    # print(D)
    x = D[:,:2]
    y = D[:,2]
    return x,y

def Q2_1_test():
    test_x = []
    for i in np.arange(-2, 2, 0.2):
        for j in np.arange(-2, 2, 0.2):
            test_x.append([i,j])
    return test_x

def Q2_1():
    x,y = load("D2z.txt")
    x10 = []
    x11 = []
    x20 = []
    x21 = []
    for i in range(y.size):
        if y[i] == 0:
            x10.append(x[i][0])
            x20.append(x[i][1])
        else:
            x11.append(x[i][0])
            x21.append(x[i][1])
    plt.scatter(x10,x20, color = "black", marker = "o", facecolors='none')
    plt.scatter(x11,x21, color = "black", marker = "+", s = 50)

    model = neighbors.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    model.fit(x,y)
    
    test_x = Q2_1_test()
    test_y = model.predict(test_x)

    test_x10 = []
    test_x20 = []
    test_x11 = []
    test_x21 = []
    for i in range(test_y.size):
        if test_y[i] == 0:
            test_x10.append(test_x[i][0])
            test_x20.append(test_x[i][1])
        else:
            test_x11.append(test_x[i][0])
            test_x21.append(test_x[i][1])
    plt.scatter(test_x10, test_x20, color = "blue")
    plt.scatter(test_x11, test_x21, color = "red")

    plt.title("Q2-1 sol")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    plt.show()

if __name__ == "__main__":
    Q2_1()