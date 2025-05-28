import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def least_diagonally_dominant(H):
    H = np.array(H)
    n = H.shape[0]
    diffs = []
    for i in range(n):
        diag = H[i, i]
        off_diag_sum =  np.sum(np.abs(H[i, :])) - abs(H[i, i])
        delta = diag - off_diag_sum
        diffs.append(delta)

    # Find the row with the least diffs
    print(diffs)
    min_delta = min(diffs)

    return min_delta


# Training with gradient descent
def train_logistic_regression(A, b, lambd=1e-3, lr=0.1, max_iter=10000, tol=1e-8, print_cost = False):
    x = np.ones(A.shape[1])
    costs = []

    for i in range(max_iter):
        grad = logistic_regression_gradient(x, A, b, lambd)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"Gradient norm below threshold at iteration {i}. Stopping early.")
            break
        x = x - lr * grad # gradient descent
        cost = logistic_regression_cost(x, A, b, lambd)
        costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost (iteration %i) = %f" %(i, cost))
    return x, costs

# Accuracy function
def accuracy(x, A, b):
    preds = np.sign(A @ x)
    return np.mean(preds == b)

# Load MNIST
mnist = fetch_openml('mnist_784', version=1)
X_all = mnist['data'].to_numpy().astype(np.float32) / 255.0  # (70000, 784)
y_all = mnist['target'].astype(int)

# Binary classification: digits 0 and 1
mask = (y_all == 0) | (y_all == 1)
A = X_all[mask]                     # shape: (N, 784)
b = np.where(y_all[mask] == 0, -1, 1)  # shape: (N,)

# Train/Test split (e.g., 80/20)
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

# Train
x_opt, costs = train_logistic_regression(A_train, b_train, lambd=1e-5, lr=0.5, max_iter=10000,tol=1e-8)

# Plot loss curve
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

# Evaluate
train_acc = accuracy(x_opt, A_train, b_train)
test_acc = accuracy(x_opt, A_test, b_test)
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy:  {test_acc * 100:.2f}%")

# Initialize optimization variable x
x = np.ones(A.shape[1])  # shape: (784,)

Hess = logistic_regression_hessian(x,A,b,1e-3)

lambd = least_diagonally_dominant(Hess)
print(lambd)


## Parameters
N = 56 # Agent number
comDelay = 0
upDelay = 1
Xmax = 10
Xmin = 1
