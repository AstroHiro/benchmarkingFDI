import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap='coolwarm')
plt.show
print("test")