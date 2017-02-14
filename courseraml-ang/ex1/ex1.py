import numpy as np
import matplotlib.pyplot as plt
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
theta, J_history = gradientDescent.compute(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: {}, {}'.format(theta[0], theta[1]))

# plot the histrory of J value
plt.plot(J_history)
plt.show()

# plot result
plotData.plotResult(X, y, theta)

# Predict values for population sizes of 35,000 and 70,000
# predict1 = np.array([1,3.5]).dot(theta)
predict1 = theta.T.dot([1,3.5])
print('For population = 35,000, we predict a profit of {}'.format(predict1*10000))

# predict2 = np.array([1, 7]).dot(theta)
predict2 = theta.T.dot([1,7])
print('For population = 70,000, we predict a profit of {}'.format(predict2*10000))


## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')
