import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
y_sigmoid = []
for i in range(len(random_values)):
    y_sigmoid.append(sigmoid(random_values[i]))  # Append the sigmoid value to the list

# Visualize
plt.scatter(random_values, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
