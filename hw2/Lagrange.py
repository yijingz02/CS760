import numpy as np
from scipy.interpolate import lagrange
import math
from numpy import polyval

def noise(x,std):
    n = np.random.normal(0, std, size = x.shape)
    return x + n

def generate(a, b, s):
    x = np.empty(0)
    y = np.empty(0)
    for i in range(s):
        tmp = np.random.uniform(low=a, high=b, size=None)
        x = np.append(x, tmp)
        y = np.append(y, math.sin(tmp))
    return x, y

    
def calc_error(poly, x, y):
    err = 0
    for i, n in enumerate(x):
        err += abs(y[i] - poly(n))
    return err / x.size


def calc_noisedy(x):
    y = np.empty(0)
    for n in x:
        y = np.append(y, math.sin(n))
    return y

if __name__ == "__main__":
    x_train, y_train = generate(-1, 1, 100)
    x_test,  y_test  = generate(-1, 1, 100)
    
    poly = lagrange(x_train, y_train)
    
    print("<-- Train Error without noise -->")
    print("Err:", calc_error(poly, x_train, y_train))

    print("<-- Test Error without noise -->")
    print("Err:", calc_error(poly, x_test, y_test))

    print("<-- Test Error with noise -->")
    stds = [0.1, 0.3, 0.5, 0.7, 0.9]

    for n in stds:
        print("std:", n)
        x_noise = noise(x_train, n)
        y_noise = calc_noisedy(x_noise)
        poly_noise = lagrange(x_noise, y_noise)
        print("Err:", calc_error(poly_noise, x_test, y_test))

    calc_error(poly, x_train, y_train)
