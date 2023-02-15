import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
import matplotlib.pyplot as plt

def load(filename):
    D = pd.read_csv(filename, sep = " ", header = None)
    D.columns = ['x1', 'x2', 'y']
    D[D.columns] = D[D.columns].apply(pd.to_numeric, errors='coerce')
    return D

if __name__ == "__main__":

    err = []

    Dtest = load("Dtest.txt")

    print("<-- D8192 -->")
    D8192 = load("D8192.txt")
    t = DecisionTreeClassifier()
    t = t.fit(D8192[['x1','x2']],D8192['y'])
    y_pred = t.predict(Dtest[['x1','x2']])
    tmp = 1 - metrics.accuracy_score(Dtest['y'], y_pred)
    err.append(tmp)
    print("Nodes: ", t.tree_.node_count)
    print("Error: ", tmp)

    print("<-- D2048 -->")
    D2048 = load("D2048.txt")
    t = DecisionTreeClassifier()
    t = t.fit(D2048[['x1','x2']], D2048['y'])
    y_pred = t.predict(Dtest[['x1','x2']])
    tmp = 1 - metrics.accuracy_score(Dtest['y'], y_pred)
    err.append(tmp)
    print("Nodes: ", t.tree_.node_count)
    print("Error: ", tmp)

    print("<-- D512 -->")
    D512 = load("D512.txt")
    t = DecisionTreeClassifier()
    t = t.fit(D512[['x1','x2']],D512['y'])
    y_pred = t.predict(Dtest[['x1','x2']])
    tmp = 1 - metrics.accuracy_score(Dtest['y'], y_pred)
    err.append(tmp)
    print("Nodes: ", t.tree_.node_count)
    print("Error: ", tmp)

    print("<-- D128 -->")
    D128 = load("D128.txt")
    t = DecisionTreeClassifier()
    t = t.fit(D128[['x1','x2']],D128['y'])
    y_pred = t.predict(Dtest[['x1','x2']])
    tmp = 1 - metrics.accuracy_score(Dtest['y'], y_pred)
    err.append(tmp)
    print("Nodes: ", t.tree_.node_count)
    print("Error: ", tmp)

    print("<-- D32 -->")
    D32 = load("D32.txt")
    t = DecisionTreeClassifier()
    t = t.fit(D32[['x1','x2']],D32['y'])
    y_pred = t.predict(Dtest[['x1','x2']])
    tmp = 1 - metrics.accuracy_score(Dtest['y'], y_pred)
    err.append(tmp)
    print("Nodes: ", t.tree_.node_count)
    print("Error: ", tmp)

    x = [8192, 2048, 512, 128, 32]
    plt.title('Learning curve')
    plt.xlabel('n')
    plt.ylabel('err')
    plt.plot(x, err)
    plt.show()