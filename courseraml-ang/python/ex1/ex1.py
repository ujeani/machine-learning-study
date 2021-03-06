import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = y.size
    J = 0.0
    h = X.dot(theta)
    J = np.sum(np.square(h-y))/(2*m)
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    m =  y.size
    J_history = np.zeros((iterations, 1))

    for iter in range(iterations):  # repeat for all iterations
        h = X.dot(theta)
        theta = theta - (alpha*X.T.dot(h-y))/m
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history

## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')

data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',')

X = data[:, [0]]
y = data[:, [1]]
m = data.shape[0]   # number of training examples

# Plot Data
plt.plot(X, y, 'rx')
plt.title('MarkerSize')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.show()

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
X = np.c_[np.ones(m), X] # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
print(computeCost(X, y, theta))

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ',theta[0], theta[1])

# plot the histrory of J value
plt.plot(J_history)
plt.show()

# plot result
plt.title('MarkerSize')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.plot(X[:, 1], y, 'rx')
plt.plot(X[:, 1], X.dot(theta))
plt.show()

# Predict values for population sizes of 35,000 and 70,000
# predict1 = np.array([1,3.5]).dot(theta)
predict1 = theta.T.dot([1,3.5])
print('For population = 35,000, we predict a profit of', predict1*10000)

# predict2 = np.array([1, 7]).dot(theta)
predict2 = theta.T.dot([1,7])
print('For population = 70,000, we predict a profit of ', predict2*10000)


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
