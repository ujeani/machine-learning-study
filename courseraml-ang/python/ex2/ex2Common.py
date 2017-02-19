import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1+np.exp(-z))


def mapFeature(X1, X2):
    m = X1.shape[0]
    degree = 6
    out = np.ones([m, 1])
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.concatenate((out, np.power(X1, i-j)*np.power(X2, j)), axis=1)
    return out

def plotTrainingData(X, y, xlabel="", ylabel=""):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    axes = plt.gca()
    axes.set_xlim([-1, 1.5])
    axes.set_ylim([-1, 1.5])
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'yo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return

def plotData(X, y, xlabel, ylabel):
    plotTrainingData(X, y, xlabel, ylabel)
    plt.show()
    return

def plotDecisionBoundary(theta, X, y, xlabel="", ylabel=""):
    plotTrainingData(X[:, [1, 2]], y, xlabel, ylabel)
    if X.shape[1] <=3 :
        plot_x = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        plot_y = (-1. / theta[2]) * (theta[0] + (theta[1] * plot_x))
        # print(plot_x)
        # print(plot_y)
        plt.plot(plot_x, plot_y)
    else :
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros([u.size, v.size])
        # Evaluate z = theta * x over the grid
        for i in range(z.shape[0]) :
            for j in range(z.shape[1]) :
                z[i, j] = mapFeature(np.array([[u[i]]]), np.array([[v[j]]])).dot(theta)
        z = z.T
        plt.contour(u, v, z, levels=[0])
    plt.show()
    return

def predict(theta, X):
    m = X.shape[0]
    p = sigmoid(X.dot(theta))

    for i in range(m):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p

