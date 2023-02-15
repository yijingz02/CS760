import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import math
from numpy import polyval

a = -1
b = 1

x_new = np.arange(a, b, 0.1)
fig, axs = plt.subplots(2,3)

x_train = np.empty(0)
y_train = np.empty(0)
for i in range(100):
    tmp = np.random.uniform(low=a, high=b, size=None)
    # print(tmp, math.sin(tmp))
    x_train = np.append(x_train, tmp)
    y_train = np.append(y_train, math.sin(tmp))

poly = lagrange(x_train, y_train)
print(Polynomial(poly.coef[::-1]).coef)

# axs[0,0].set_xlim(a,b)
# axs[0,0].set_ylim(-1,1)
# axs[0,0].scatter(x_train, y_train, s=10)
# axs[1,0].set_xlim(a,b)
# axs[1,0].set_ylim(-1,1)
# axs[1,0].plot(x_new, Polynomial(poly.coef[::-1])(x_new))

train_err = 0
for i, x in enumerate(x_train):
    predicted = poly(x)
    train_err += abs(predicted - y_train[i])
train_err = train_err / 100
print("Train error:", train_err)

test_err = 0
x_test = np.empty(0)
y_test = np.empty(0)
for i in range(100):
    tmp = np.random.uniform(low=a, high=b, size=None)
    # print(tmp, math.sin(tmp))
    x_test = np.append(x_test,tmp)
    predicted = poly(tmp)
    true_y = math.sin(tmp)
    y_test = np.append(y_test, true_y)
    test_err += abs(tmp - true_y)
test_err = test_err / 100
print("Test error:", test_err)

# axs[0,1].set_xlim(a,b)
# axs[0,1].set_ylim(-1,1)
# axs[0,1].scatter(x_train, y_train, s=10)
# axs[1,1].set_xlim(a,b)
# axs[1,1].set_ylim(-1,1)
# axs[1,1].plot(x_new, Polynomial(poly.coef[::-1])(x_new))

plt.show()
