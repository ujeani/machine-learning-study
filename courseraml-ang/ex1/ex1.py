import numpy as np
import plotData
import cost
import gradientDescent

## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')

data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',')

X = data[:, [0]]
y = data[:, [1]]
m = data.shape[0]   # number of training examples

# Plot Data
plotData.plot(X, y)

## =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')
X = np.c_[np.ones(m),data[:, [0]]] # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
print(cost.compute(X, y, theta))

# run gradient descent
theta, _ = gradientDescent.compute(X, y, theta, alpha, iterations)