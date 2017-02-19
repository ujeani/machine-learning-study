import numpy as np
import matplotlib.pyplot as plt
import ex2Common as common


def costFunctionReg(theta, X, y, lambda_v):
    m = y.size
    J = 0
    grad = np.zeros(np.shape(theta))

    h = common.sigmoid(X.dot(theta))
    J = -1*(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))/m
    J = J + (lambda_v/(2*m))*(theta.T.dot(theta)) - (theta[0]*theta[0]) # theta[0]는 regularization을 하지 않는다

    grad = (1/m)*(X.T.dot(h-y))+((lambda_v/m)*theta)
    grad[0] = grad[0]-((lambda_v/m)*theta[0])
    return J, grad


def gradientDescent(X, y, theta, lambda_v, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    grad = np.zeros(np.shape(theta))
    for i in range(num_iters):
        J_history[i], grad = costFunctionReg(theta, X, y, lambda_v)
        theta = theta - (alpha * grad)
        if i%10000 == 0:
            print('J(',i,')=', J_history[i, 0])
    return theta, J_history



data = np.loadtxt('ex2data2.txt', dtype=float, delimiter=',')

X = data[:, [0,1]]
y = data[:, [2]]

common.plotData(X, y, 'Microchip Test 1', 'Microchip Test 2')


## =========== Part 1: Regularized Logistic Regression ============
# In this part, you are given a dataset with data points that are not
# linearly separable. However, you would still like to use logistic
# regression to classify the data points.
#
# To do so, you introduce more features to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial
# regression).


# Add Polynomial Features


# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled

X = common.mapFeature(X[:,[0]], X[:,[1]])

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

# Initialize fitting parameters
initial_theta = np.zeros((n , 1))

# Set regularization parameter lambda to 1
lambda_v = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_v)

print('Cost at initial theta (zeros): ', cost)
print('(The cost is about  0.693)')

## ============= Part 2: Regularization and Accuracies =============
initial_theta = np.zeros((n , 1))
lambda_v = 1
alpha = 0.01
num_iters = 100000
theta, J_history = gradientDescent(X, y, initial_theta, lambda_v, alpha, num_iters)

# Plot the convergence graph
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Print theta to screen
print('theta: ', theta)

common.plotDecisionBoundary(theta, X, y, 'Microchip Test 1', 'Microchip Test 2')


# Compute accuracy on our training set
p = common.predict(theta, X)

print('Train Accuracy: ', np.mean(np.double(p == y)) * 100)


