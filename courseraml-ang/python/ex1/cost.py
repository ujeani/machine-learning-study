import numpy as np

def compute(X, y, theta):
    m = y.size
    J = 0.0
    h = X.dot(theta)
    J = np.sum(np.square(h-y))/(2*m)
    return J
