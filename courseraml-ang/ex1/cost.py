import numpy as np

def compute(X, y, theta):
    m = y.shape[0]
    J = 0
    for i in range(m):
        J = J + pow((np.dot(X[i,:], theta) - y[i]), 2)
    J = J/(2*m)
    return J
