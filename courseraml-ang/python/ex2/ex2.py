import numpy as np
import matplotlib.pyplot as plt
import ex2Common as common

def costFunction(theta, X, y):
    m = y.size
    h = common.sigmoid(X.dot(theta))
    J = -1*(y.T.dot(np.log(h))+(1-y).T.dot(np.log(1-h)))/m
    # grad = (X.T.dot(h-y))/m
    return J

def gradient(theta, X, y):
    m = y.size
    h = common.sigmoid(X.dot(theta))
    grad = (X.T.dot(h-y))/m
    return grad

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        J_history[i] = costFunction(theta, X, y)
        grad = gradient(theta, X, y)
        theta = theta - (alpha * grad)
        if i%10000 == 0:
            print('J(',i,')=', J_history[i, 0])
    return theta, J_history


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', dtype=float, delimiter=',')

X = data[:, [0,1]]
y = data[:, [2]]


## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
common.plotData(X, y, 'Exam 1 score', 'Exam 2 score')

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

# Add intercept term to x and X_test
X = np.c_[np.ones(m), X] # Add a column of ones to x

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros): ', cost)
print('Gradient at initial theta (zeros): ', grad)

## ============= Part 3: Optimizing  =============
alpha = 0.002
num_iters = 1000000
theta, J_history = gradientDescent(X, y, initial_theta, alpha, num_iters)
# Print theta to screen
print('theta: ', theta)

# Plot the convergence graph
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

common.plotDecisionBoundary(theta, X, y, 'Exam 1 score', 'Exam 2 score')



## ============== Part 4: Predict and Accuracies ==============


#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

v = np.array([1, 45, 85])
prob = common.sigmoid(v.dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of ',prob)
print('Expect to see 0.776')

# Compute accuracy on our training set
p = common.predict(theta, X)

print('Train Accuracy: ', np.mean(np.double(p == y)) * 100)

