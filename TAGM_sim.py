import numpy as np
from sklearn.datasets import fetch_openml

def logistic_regression_cost(x, A, b, lambd):
    """
    Compute logistic regression cost with L2 regularization.
    Inputs:
        x     : (d,) parameter vector
        A     : (n, d) data matrix, n is the sample number, d is the dimension of features
        b     : (n,) labels 
        lambd : regularization parameter (scalar)
    Output:
        cost  : scalar
    """
    n = A.shape[0]
    cost = 1/n * np.sum(np.log(1 + np.exp(-b * (A @ x)))) + (lambd / 2) * np.linalg.norm(x)**2

    return cost

def logistic_regression_gradient(x, A, b, lambd):
    """
    Compute gradient of logistic regression cost.
    Inputs:
        x     : (d,) parameter vector
        A     : (n, d) data matrix, n is the sample number, d is the dimension of features
        b     : (n,) labels 
        lambd : regularization parameter (scalar)
    Output:
        cost  : scalar
    """
    n = A.shape[0]
    grad = -1/n * np.sum(b * A.T / (1 + np.exp(b * (A @ x))),axis=1) + lambd * x

    return grad

def logistic_regression_hessian(x, A, b, lambd):
    """
    Compute Hessian of logistic regression cost.
    """
    n, d = A.shape
    H = np.zeros((d, d))
    for i in range(n):
        a_i = A[i, :].reshape(-1, 1)  # column vector
        z_i = b[i] * (a_i.T @ x)
        coeff =  b[i]**2 * np.exp(z_i) / (1 + np.exp(z_i))**2
        H += coeff * (a_i @ a_i.T)
    H = H / n + lambd * np.eye(d)

    return H

# Load MNIST
mnist = fetch_openml('mnist_784', version=1)
X_all = mnist['data'].to_numpy().astype(np.float32) / 255.0  # (70000, 784)
y_all = mnist['target'].astype(int)

# Binary classification: digits 0 and 1
mask = (y_all == 0) | (y_all == 1)
A = X_all[mask]                     # shape: (N, 784)
b = np.where(y_all[mask] == 0, -1, 1)  # shape: (N,)

# Initialize optimization variable x
x = np.zeros(A.shape[1])  # shape: (784,)

Hess = logistic_regression_hessian(x,A,b,0)





## Parameters
N = 10 # Agent number
comDelay = 0
upDelay = 1
Xmax = 10
Xmin = 1
