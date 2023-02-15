import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def load(filename):
    D = pd.read_csv(filename, sep = " ", header = None)
    D.columns = ['x1', 'x2', 'y']
    D[D.columns] = D[D.columns].apply(pd.to_numeric, errors='coerce')
    return D

class Node():
    def __init__(self, feature, threshold):
        self.left = None
        self.right = None
        self.predict_label = None
        self.feature = feature
        self.threshold = threshold
        self.gainRatio = None

class decision_tree():
    def __init__(self, D, features, prediction):
        self.D = D
        self.features = features
        self.prediction = prediction

    def MakeSubtree(self, D):
        cnt0 = D[D[self.prediction] == 0].shape[0]
        cnt1 = D[D[self.prediction] == 1].shape[0]

        if cnt0 != 0 and cnt1 != 0:
            gain, f, t = self.DetermineCandidateSplits(D)

        # print(f, t)

        if cnt0 == 0 or cnt1 == 0 or (gain == 0): # Make a leaf 
            tmp  =  Node(None, None)
            if cnt0 <= cnt1:
                tmp.predict_label = 1
            else:
                tmp.predict_label = 0

            return tmp

        else: # make an internal node
            tmp = Node(f,t)

            l = D[D[f] >= t]
            r = D[D[f] <  t]

            tmp.left  = self.MakeSubtree(l)
            tmp.right = self.MakeSubtree(r)
            tmp.predict_label = None

            return tmp
            
    def DetermineCandidateSplits(self, D):
        gain = -1
        feature = None
        threshold = None

        p0 = D[D[self.prediction] == 1]
        p1 = D[D[self.prediction] == 0]

        py0 = p0.shape[0] / (p0.shape[0]+p1.shape[0])
        py1 = p1.shape[0] / (p0.shape[0]+p1.shape[0])

        hy = 0
        if py0 != 0:
            hy -= py0 * math.log2(py0)
        if py1 != 0:
            hy -= py1 * math.log2(py1)

        for f in self.features:
            for t in D[f]:
                
                cnt_left0 = p0[p0[f] >= t].shape[0]
                cnt_left1 = p1[p1[f] >= t].shape[0]

                cnt_right0 = p0[p0[f] < t].shape[0]
                cnt_right1 = p1[p1[f] < t].shape[0]

                if (cnt_left0 + cnt_left1) == 0:
                    hyl = 0
                else:
                    pyl0 = cnt_left0 / (cnt_left0 + cnt_left1)
                    pyl1 = cnt_left1 / (cnt_left0 + cnt_left1)
                    hyl = 0
                    if pyl0 != 0:
                        hyl -= pyl0 * math.log2(pyl0)
                    if pyl1 != 0:
                        hyl -= pyl1 * math.log2(pyl1)

                if (cnt_right0 + cnt_right1) == 0:
                    hyr = 0
                else:
                    pyr0 = cnt_right0 / (cnt_right0 + cnt_right1)
                    pyr1 = cnt_right1 / (cnt_right0 + cnt_right1)
                    hyr = 0
                    if pyr0 != 0:
                        hyr -= pyr0 * math.log2(pyr0)
                    if pyr1 != 0:
                        hyr -= pyr1 * math.log2(pyr1)

                pl = (cnt_left0  + cnt_left1)  / (p0.shape[0]+p1.shape[0])
                pr = (cnt_right0 + cnt_right1) / (p0.shape[0]+p1.shape[0])

                hys = pl * hyl + pr * hyr

                infoGain = hy - hys

                if infoGain == 0:
                    gainRatio = 0
                else:
                    hs = 0
                    if pl != 0:
                        hs -= pl * math.log2(pl)
                    if pr != 0:
                        hs -= pr * math.log2(pr)

                    if hs != 0:
                        gainRatio = infoGain / hs
                    else:
                        gainRatio = infoGain

                # print(f,t,gainRatio)

                if gainRatio > gain:
                    gain = gainRatio
                    feature = f
                    threshold = t

        return gain, feature, threshold
    
    def printTree(self, r):
        queue = []
        queue.append((r,0))

        prev = 0
        s = ""

        while(len(queue) != 0):
            (tmp,l) = queue.pop(0)

            if l != prev:
                if l == 0:
                    continue
                print("Level " + str(l) + ": " + s)
                s = ""
                prev = l

            if tmp.predict_label != None:
                s += ("(y: " + str(tmp.predict_label) + "), ")
                # print("Leaf: " + str(tmp.predict_label) + ", ")
                continue
            
            s += ("(" + tmp.feature + "," + str(tmp.threshold) + "), ")
            # print(tmp.feature + ":" + str(tmp.threshold) + ", ")

            if tmp.left != None:
                queue.append((tmp.left, l+1))
            if tmp.right != None:
                queue.append((tmp.right,l+1))

        print("Level " + str(prev+1) + ": " + s)

    def count_nodes(self, r):
        queue = []
        queue.append(r)

        cnt = 0

        while(len(queue) != 0):
            tmp = queue.pop(0)
            cnt += 1
            if tmp.left != None:
                queue.append(tmp.left)
            if tmp.right != None:
                queue.append(tmp.right)

        return cnt

    def test_error(self, test, r):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        test_array = test.to_numpy()
        for x in test_array:
            if self.predict(r,x) != x[2]:
                if x[2] == 0:
                    FN += 1
                else:
                    FP += 1
            else:
                if x[2] == 0:
                    TN += 1
                else:
                    TP += 1

        err = (FN+FP)/(TP+FP+TN+FN)
        return err
            
    def predict(self, r, x):
        if r.predict_label != None:
            return r.predict_label
        
        if r.feature == "x1":
            f = 0
        else:
            f = 1

        if x[f] >= r.threshold:
            return self.predict(r.left, x)
        else:
            return self.predict(r.right,x)

    def plot_decision_boundry(self, ax, r, x1l, x1h, x2l, x2h):
        # print(x1l, x1h, x2l, x2h)
        if r.predict_label != None:
            if r.predict_label == 1:
                tmp = plt.Rectangle((x1l, x2l),(x1h-x1l),(x2h-x2l), color='red',  alpha = 0.3, lw=0)
                ax.add_patch(tmp)
            if r.predict_label == 0:
                tmp = plt.Rectangle((x1l, x2l),(x1h-x1l),(x2h-x2l), color='blue', alpha = 0.3, lw=0)
                ax.add_patch(tmp)
            return

        if r.left != None:
            if r.feature == "x1":
                self.plot_decision_boundry(ax, r.left, r.threshold, x1h, x2l, x2h)
            else:
                self.plot_decision_boundry(ax, r.left, x1l, x1h, r.threshold, x2h)
        if r.right != None:
            if r.feature == "x1":
                self.plot_decision_boundry(ax, r.right, x1l, r.threshold, x2l, x2h)
            else:
                self.plot_decision_boundry(ax, r.right, x1l, x1h, x2l, r.threshold)

        return
    
    def plot_graph(self, D, r, ax, l,h):
        label0_x1 = []
        label0_x2 = []
        label1_x1 = []
        label1_x2 = []
        tmp = D.to_numpy()
        for i in tmp:
            if i[2] == 0:
                label0_x1.append(i[0])
                label0_x2.append(i[1])
            else:
                label1_x1.append(i[0])
                label1_x2.append(i[1])
        ax.set(xlabel='x1', ylabel='x2')
        ax.scatter(label0_x1, label0_x2, s=20)
        ax.scatter(label1_x1, label1_x2, s=20)
        self.plot_decision_boundry(ax, r, l, h, l, h)

def Q2_7():
    D = load("Dbig.txt")
    # print(D)

    y_err = []
    fig, axs = plt.subplots(2, 3)

    print("<-- D8192 -->")
    D8192 = D.sample(n = 8192)
    D8192.to_csv("D8192.txt", sep = " ", header = None, index=False)
    test = D.drop(D8192.index)
    test.to_csv("Dtest.txt", sep = " ", header = None, index=False)
    t = decision_tree(D8192,['x1','x2'],'y')
    r = t.MakeSubtree(D8192)
    print("Finished making tree")
    # t.printTree(r)
    err = t.test_error(test,r)
    print("nodes: " + str(t.count_nodes(r)))
    print("error: " + str(err))
    y_err.append(err)
    axs[0, 1].set_title('D8192')
    t.plot_graph(D8192, r, axs[0,1], -1.6, 1.6)

    print("<-- D2048 -->")
    D2048 = D8192.head(2048)
    D2048.to_csv("D2048.txt", sep = " ", header = None, index=False)
    t = decision_tree(D2048,['x1','x2'],'y')
    r = t.MakeSubtree(D2048)
    print("Finished making tree")
    # t.printTree(r)
    err = t.test_error(test,r)
    print("nodes: " + str(t.count_nodes(r)))
    print("error: " + str(err))
    y_err.append(err)
    axs[0, 2].set_title('D2048')
    t.plot_graph(D2048, r, axs[0,2], -1.6, 1.6)

    print("<-- D512 -->")
    D512 = D8192.head(512)
    D512.to_csv("D512.txt", sep = " ", header = None, index=False)
    t = decision_tree(D512,['x1','x2'],'y')
    r = t.MakeSubtree(D512)
    print("Finished making tree")
    # t.printTree(r)
    err = t.test_error(test,r)
    print("nodes: " + str(t.count_nodes(r)))
    print("error: " + str(err))
    y_err.append(err)
    axs[1, 0].set_title('D512')
    t.plot_graph(D512, r, axs[1,0], -1.6, 1.6)

    print("<-- D128 -->")
    D128 = D8192.head(128)
    D128.to_csv("D128.txt", sep = " ", header = None, index=False)
    t = decision_tree(D128,['x1','x2'],'y')
    r = t.MakeSubtree(D128)
    print("Finished making tree")
    # t.printTree(r)
    err = t.test_error(test,r)
    print("nodes: " + str(t.count_nodes(r)))
    print("error: " + str(err))
    y_err.append(err)
    axs[1, 1].set_title('D128')
    t.plot_graph(D128, r, axs[1,1], -1.6, 1.6)

    print("<-- D32 -->")
    D32 = D8192.head(32)
    D32.to_csv("D32.txt", sep = " ", header = None, index=False)
    t = decision_tree(D32,['x1','x2'],'y')
    r = t.MakeSubtree(D32)
    print("Finished making tree")
    # t.printTree(r)
    err = t.test_error(test,r)
    print("nodes: " + str(t.count_nodes(r)))
    print("error: " + str(err))
    y_err.append(err)
    axs[1, 2].set_title('D32')
    t.plot_graph(D32, r, axs[1,2], -1.6, 1.6)

    print("<-- start plotting -->")

    x = [8192, 2048, 512, 128, 32]
    axs[0, 0].plot(x, y_err)
    axs[0, 0].set_title('Learning curve')
    axs[0, 0].set(xlabel='n', ylabel='err')
    plt.show()

def D1():
    fig, axs = plt.subplots()
    D1 = load("D1.txt")
    t = decision_tree(D1,['x1','x2'],'y')
    r = t.MakeSubtree(D1)
    axs.set_title('D1.txt')
    t.plot_graph(D1, r, axs, 0,1)
    plt.show()

if __name__ == "__main__":
    D1()
    # Q2_7()