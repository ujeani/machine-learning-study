import numpy as np
import cost

def compute(X, y, theta, alpha, iterations):
    m =  y.size
    J_history = np.zeros((iterations, 1))

    for iter in range(iterations):  # repeat for all iterations
        h = X.dot(theta)
        theta = theta - (alpha*X.T.dot(h-y))/m
        J_history[iter] = cost.compute(X, y, theta)

    return theta, J_history