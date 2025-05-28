import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Define the projection argument for projected gradient descent
def projection(arr, lower_bound, upper_bound):
    return np.clip(np.asarray(arr), lower_bound, upper_bound)

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
    
    # print(diffs)
    # Find the row with the least diffs
    min_delta = min(diffs)

    return min_delta


# Training with gradient descent
def train_logistic_regression(A, b, lambd=1e-3, lr=1e-3, max_iter=1000, tol=1e-6, print_cost = False):
    x = np.ones(A.shape[1])
    costs = []
    cost = logistic_regression_cost(x, A, b, lambd)
    costs.append(cost)

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

##################################################################################################################

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

# Initialize optimization variable x
x = np.zeros(A.shape[1])  # shape: (784,)
# Hessian checker 
Hess = logistic_regression_hessian(x,A,b,1e-3)
lambd = least_diagonally_dominant(Hess)
# print(lambd)
# print("Hessian:")
# print(Hess)

# Train
gamma = 0.1/ np.max(np.abs(Hess))
print("gamma is:")
print(gamma)
x_star, costs = train_logistic_regression(A_train, b_train, lambd=1e-3, lr=gamma, max_iter=500,tol=1e-8)

# # Evaluate
train_acc = accuracy(x_star, A_train, b_train)
test_acc = accuracy(x_star, A_test, b_test)
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy:  {test_acc * 100:.2f}%")

print("True state:")
print(x_star)

# # Plot loss curve
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

##############################################################################################################################


####################################################### Algorithm ############################################################
# ## Parameters for distributed 
# N = 56 # Agent number
# comDelay = 0
# upDelay = 1
# lb = 0
# ub = 10

# # Neighbor sets
# neigh = np.tile(np.arange(1, N + 1).reshape(-1, 1), (1, N))


# #  # Create instances of the Agent classes
# block_size = 784 // N

# ## Initial state
# x0 = np.ones((1, 784, N))
# y0 = np.ones((1, 784, N))

# # ## Hyperparameters
# # gammaHB = gammaNAG = gammaGM = gamma
# # mu = 1e-3
# # lambdaHB = (gammaNAG*mu)/2
# # lambdaNAG = (gammaNAG*mu)/(2*(1-gammaNAG*mu))
# # lambdaGM =(gammaGM*mu)/(2*(1-gammaGM*mu)); 
# # betaGM =  (gammaGM*mu)/(2*(1-gammaGM*mu)); 

# dt = 0.1
# tf = 60
# tvec = np.arange(0, tf + dt, dt)

# # initialization
# trueStateGD = minusMinGD = np.zeros((14, 56))
# xvecGD = np.zeros((len(tvec)+1, 784, N))

# M, d = A_train.shape        # Total samples and feature dimension

# # Split the data equally (or as equally as possible)
# data_per_agent = M // N
# # print(data_per_agent)
# remainder = M % N

# A_train_list = []
# b_train_list = []

# start_idx = 0
# for i in range(N):
#     # Distribute the remainder (M % N) over the first few agents
#     num_samples = data_per_agent + (1 if i < remainder else 0)
#     end_idx = start_idx + num_samples
    
#     A_i = A_train[start_idx:end_idx, :]   # shape: (num_samples, d)
#     b_i = b_train[start_idx:end_idx]      # shape: (num_samples,)

#     A_train_list.append(A_i)
#     b_train_list.append(b_i)

#     start_idx = end_idx

# probs = [1, 0.5, 0.1]

# if upDelay == 1:
#     for prob in probs:
#         probCom = prob * np.ones(N)

#         opsk = 0
#         # xvecGD = x0.copy()
#         yvecGD = y0.copy()
#         xvecNAG = x0.copy()
#         yvecNAG = y0.copy()
#         xvecHB = x0.copy()
#         yvecHB = y0.copy()
#         xvecGM = x0.copy()
#         yvecGM = y0.copy()

#         for k in range(len(tvec)):
#             upif = np.where(np.random.rand(N) < probCom)[0] if upDelay == 1 else np.arange(N)

#             for n in range(N):
#                 xvecGD[k+1,:,n] = xvecGD[k,:,n] # copy over
#                 # xvecHB[k+1,:,n] = xvecHB[k,:,n]
#                 # yvecHB[k+1,:,n] = yvecHB[k,:,n]
#                 # xvecNAG[k+1,:,n] = xvecNAG[k,:,n]
#                 # yvecNAG[k+1,:,n] = yvecNAG[k,:,n]
#                 # xvecGM[k+1,:,n] = xvecGM[k,:,n]
#                 # yvecGM[k+1,:,n] = yvecGM[k,:,n]

#             if len(upif) > 0:
#                 for cag in upif:
#                     # cag is the current agent index
#                     idx_start = N * cag - (N - cag)
#                     idx_end = N * cag

#                     # GD
#                     xcur = xvecGD[k,:,cag]
#                     x_block = xcur[cag * d : (cag + 1) * d].reshape(-1, 1)  # agent `cag`'s local parameters
#                     # A_i = A_train_list[cag]       # shape: (m_i, d)
#                     # b_i = b_train_list[cag]       # shape: (m_i,)
#                     gradif = logistic_regression_gradient(xcur.reshape(-1, 1), A_train, b_train, 1e-3)
#                     xnew_block = projection(x_block - gamma * gradif,lb,ub)
#                     xvecGD[k+1,:,cag] = xcur
#                     xvecGD[k + 1, cag * d : (cag + 1) * d, cag] = xnew_block[:, 0]

#                     # # NAG
#                     # xcur = xvecNAG[-1,:,cag]
#                     # ycur = yvecNAG[-1,:,cag]
#                     # zinp = xcur + lambdaNAG * (xcur - ycur)
#                     # gradif =
#                     # ynew = projection(xcur[cag] + lambdaNAG * (xcur[cag] - ycur[cag]) - gammaNAG * gradif,lb,ub)
#                     # yvecNAG[k+1,:,cag] = ycur
#                     # yvecNAG[k+1,cag,cag] = ynew
#                     # ycur = yvecNAG[k+1,:,cag]
#                     # zinp = ycur + lambdaNAG * (ycur - xcur)
#                     # gradif = 2 * 
#                     # xnew = np.clip(ycur[cag] + lambdaNAG * (ycur[cag] - xcur[cag]) - gammaNAG * gradif, lb,ub)
#                     # xvecNAG[k+1,:,cag] = xcur
#                     # xvecNAG[k+1,cag,cag] = xnew

#                     # # HB
#                     # xcur = xvecHB[-1,:,cag]
#                     # ycur = yvecHB[-1,:,cag]
#                     # zinp = xcur
#                     # gradif = 
#                     # ynew = np.clip(xcur[cag] + lambdaHB * (xcur[cag] - ycur[cag]) - gammaHB * gradif, lb,ub)
#                     # yvecHB[k+1,:,cag] = ycur
#                     # yvecHB[k+1,cag,cag] = ynew
#                     # ycur = yvecHB[k+1,:,cag]
#                     # zinp = ycur
#                     # gradif = 
#                     # xnew = np.clip(ycur[cag] + lambdaHB * (ycur[cag] - xcur[cag]) - gammaHB * gradif, lb,ub)
#                     # xvecHB[k+1,:,cag] = xcur
#                     # xvecHB[k+1,cag,cag] = xnew

#                     # # GM
#                     # xcur = xvecGM[-1,:,cag]
#                     # ycur = yvecGM[-1,:,cag]
#                     # zinp = xcur + lambdaGM * (xcur - ycur)
#                     # gradif = 
#                     # ynew = np.clip(xcur[cag] + betaGM * (xcur[cag] - ycur[cag]) - gammaGM * gradif, lb,ub)
#                     # yvecGM[k+1,:,cag] = ycur
#                     # yvecGM[k+1,cag,cag] = ynew
#                     # ycur = yvecGM[k+1,:,cag]
#                     # zinp = ycur + lambdaGM * (ycur - xcur)
#                     # gradif = 
#                     # xnew = np.clip(ycur[cag] + betaGM * (ycur[cag] - xcur[cag]) - gammaGM * gradif, lb,ub)
#                     # xvecGM[k+1,:,cag] = xcur
#                     # xvecGM[k+1,cag,cag] = xnew

#             comif = np.where(np.random.rand(N) < probCom)[0] if comDelay == 1 else np.arange(N)

#             if len(comif) > 0:
#                 for cag in comif:
#                     sendto = np.intersect1d(np.arange(N), neigh[:,cag])
#                     sendto = sendto[sendto != cag]

#                     for ii in sendto:
#                         xvecGD[k+1,cag,ii] = xvecGD[-1,cag,cag]
#                         # xvecNAG[k+1,cag,ii] = xvecNAG[-1,cag,cag]
#                         # yvecNAG[k+1,cag,ii] = yvecNAG[-1,cag,cag]
#                         # xvecHB[k+1,cag,ii] = xvecHB[-1,cag,cag]
#                         # yvecHB[k+1,cag,ii] = yvecHB[-1,cag,cag]
#                         # xvecGM[k+1,cag,ii] = xvecGM[-1,cag,cag]
#                         # yvecGM[k+1,cag,ii] = yvecGM[-1,cag,cag]
#             else:
#                 for n in range(N):
#                     xvecGD[k+1,:,n] = xvecGD[k,:,n]
#                     # xvecNAG[k+1,:,n] = xvecNAG[k,:,n]
#                     # yvecNAG[k+1,:,n] = yvecNAG[k,:,n]
#                     # xvecHB[k+1,:,n] = xvecHB[k,:,n]
#                     # yvecHB[k+1,:,n] = yvecHB[k,:,n]
#                     # xvecGM[k+1,:,n] = xvecGM[k,:,n]
#                     # yvecGM[k+1,:,n] = yvecGM[k,:,n]
#         for i in range(N):
#             trueStateGD[:,i] = xvecGD[:,i,i]
#             minusMinGD[:,i] = np.abs(trueStateGD[:,i] - x_star[-1,0])

#             # trueStateNAG[:,i] = xvecNAG[:,i,i]
#             # minusMinNAG[:,i] = np.abs(trueStateNAG[:,i] - x_star[-1,0])

#             # trueStateHB[:,i] = xvecHB[:,i,i]
#             # minusMinHB[:,i] = np.abs(trueStateHB[:,i] - x_star[-1,0])

#             # trueStateGAG[:,i] = xvecGM[:,i,i]
#             # minusMinGAG[:,i] = np.abs(trueStateGAG[:,i] - x_star[-1,0])

#         worstPerfGD = np.max(minusMinGD, axis=1)
#         # worstPerfNAG = np.max(minusMinNAG, axis=1)
#         # worstPerfHB = np.max(minusMinHB, axis=1)
#         # worstPerfGAG = np.max(minusMinGAG, axis=1)

##############################################################################################################################