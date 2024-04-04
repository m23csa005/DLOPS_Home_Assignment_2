import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)

# Generate data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y_relu = []
y_leaky_relu = []
y_tanh = []

for i in range(len(random_values)):
    y_relu.append(relu(random_values[i]))
    y_leaky_relu.append(leaky_relu(random_values[i]))
    y_tanh.append(tanh(random_values[i]))

# Plotting

plt.scatter(random_values, y_relu, label='ReLU', color='orange')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()


plt.scatter(random_values, y_leaky_relu, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()


plt.scatter(random_values, y_tanh, label='Tanh', color='red')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()


