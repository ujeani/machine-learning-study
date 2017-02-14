import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    X_norm = X
    K=X.shape[1]
    mu = np.zeros((1, K))
    sigma = np.zeros((1, K))

    for k in range(K):
        mu[0, k] = np.mean(X[:, k])
        sigma[0, k] = np.std(X[:, k])

    for i in range(X.shape[0]):
        X_norm[i, :] = (X[i,:]-mu)/sigma

    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    J=0
    h = X.dot(theta) - y
    J = (h.T.dot(h))/(2*m)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - (alpha * X.T.dot(h - y)) / m
        J_history[i] = computeCostMulti(X, y, theta)
    return theta, J_history



def normalEqn(X, y):
    theta = np.linalg.inv(X.T.dot(X))
    theta = theta.dot(X.T.dot(y))
    return theta


## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

# Load Data
data = np.loadtxt('ex1data2.txt', dtype=float, delimiter=',')

X = data[:, [0,1]]
y = data[:, [2]]
m = y.size

# Print out some data points
print('First 10 examples from the dataset:')
for i in range(10):
    print(' X = [{}, {}], y = {}'.format(X[i, 0], X[i, 1], y[i]))


# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)

print('First 10 examples from the dataset (normalized) :')
for i in range(10):
    print(' X = [{}, {}], y = {}'.format(X[i, 0], X[i, 1], y[i]))

# Add intercept term to X
X = np.c_[np.ones(m), X] # Add a column of ones to x

# for i in range(10):
#     print(' X = [{}, {}, {}], y = {}'.format(X[i, 0], X[i, 1], X[i, 2], y[i]))

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 1000

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.xlim((0, 1000))
plt.ylim((0, 7e10))
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)


# Estimate the price of a 1650 sq-ft, 3 br house

price = theta.T.dot([1, (1650-mu[0,0])/sigma[0,0], (3-mu[0,1])/sigma[0,1]])
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):{}'.format(price))

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...')


# Load Data

X = data[:, [0,1]]
y = data[:, [2]]
m = y.size

# Add intercept term to X
X = np.c_[np.ones(m), X] # Add a column of ones to x

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: ')
print(theta)


# Estimate the price of a 1650 sq-ft, 3 br house
price = theta.T.dot([1, 1650, 3])

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): {}'.format(price))

