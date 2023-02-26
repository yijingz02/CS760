import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def load(filename):
    D = pd.read_csv(filename, sep = ",", header=0)
    y = D['Prediction'].to_numpy()
    iy = list(D.keys()).index("Prediction")

    D = D.to_numpy()
    x = np.hstack((D[:,1:iy], D[:,iy+1:]))

    return x,y

def Q2_4():
    x, y = load("emails.csv")
    ks = [1,3,5,7,10]
    avgs = []

    for k in ks:
        avg = 0
        model = neighbors.KNeighborsClassifier(n_neighbors=k)
        for i in range(5):
            s = max(0,i*1000-1)
            e = (i+1)*1000

            test_x = x[s:e,:]
            test_y = y[s:e]

            train_x = np.vstack((x[:s,:],x[e:,:]))
            train_y = np.hstack((y[:s],y[e:]))

            model.fit(train_x, train_y)
            predicted_y = model.predict(test_x)
            accuracy = accuracy_score(test_y, predicted_y)
            avg += accuracy
        
        avg /= 5
        print("k=",k,":",avg)
        avgs.append(avg)
    
    plt.title("Q2-4: kNN 5-Fold Cross validation sol")
    plt.xlabel("k")
    plt.ylabel("Average accuracy")
    plt.plot(ks, avgs, marker="o")
    plt.show()

if __name__ == "__main__":
    Q2_4()