import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    g = np.zeros(z.shape[0])
    g = 1 / (1+np.exp(-1*z))
    return g


def costFunction(theta, X, y):
    J = 0
    m = y.size
    grad = np.zeros(theta.size)

    return J, grad


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

pos = np.where(y==1)
neg = np.where(y==0)

plt.plot(X[pos, 0], X[pos, 1], 'k+')
plt.plot(X[neg, 0], X[neg, 1], 'yo')
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.show()


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

print(m, n)

# Add intercept term to x and X_test
X = np.c_[np.ones(m), X] # Add a column of ones to x

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

#
# fprintf('Cost at initial theta (zeros): %f\n', cost);
# fprintf('Gradient at initial theta (zeros): \n');
# fprintf(' %f \n', grad);
#
# fprintf('\nProgram paused. Press enter to continue.\n');
# pause;
