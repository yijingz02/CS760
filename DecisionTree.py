import hw2
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)

D1 = hw2.load("D1.txt")
t = hw2.decision_tree(D1,['x1','x2'],'y')
r = t.MakeSubtree(D1)
label0_x1 = []
label0_x2 = []
label1_x1 = []
label1_x2 = []
tmp = D1.to_numpy()
for i in tmp:
    if i[2] == 0:
        label0_x1.append(i[0])
        label0_x2.append(i[1])
    else:
        label1_x1.append(i[0])
        label1_x2.append(i[1])
axs[0, 0].scatter(label0_x1, label0_x2)
axs[0, 0].scatter(label1_x1, label1_x2)
axs[0, 0].set_title('D1.txt')
axs[0, 0].set(xlabel='x1', ylabel='x2')

axs[1, 0].scatter(label0_x1, label0_x2)
axs[1, 0].scatter(label1_x1, label1_x2)
axs[1, 0].set(xlabel='x1', ylabel='x2')
t.plot_decision_boundry(axs[1, 0], r, 0, 1, 0, 1)

D2 = hw2.load("D2.txt")
t = hw2.decision_tree(D2,['x1','x2'],'y')
r = t.MakeSubtree(D2)
label0_x1 = []
label0_x2 = []
label1_x1 = []
label1_x2 = []
tmp = D2.to_numpy()
for i in tmp:
    if i[2] == 0:
        label0_x1.append(i[0])
        label0_x2.append(i[1])
    else:
        label1_x1.append(i[0])
        label1_x2.append(i[1])
axs[0, 1].scatter(label0_x1, label0_x2)
axs[0, 1].scatter(label1_x1, label1_x2)
axs[0, 1].set_title('D2.txt')
axs[0, 1].set(xlabel='x1', ylabel='x2')

axs[1, 1].scatter(label0_x1, label0_x2)
axs[1, 1].scatter(label1_x1, label1_x2)
axs[1, 1].set(xlabel='x1', ylabel='x2')
t.plot_decision_boundry(axs[1, 1], r, 0, 1, 0, 1)

plt.show()