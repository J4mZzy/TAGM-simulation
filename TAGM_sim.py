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

        # print("Gradient is:")
        # print(grad)
        
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
lambd_bound = least_diagonally_dominant(Hess)
lambd = 1e-3 # Regularization parameter 
# print(lambd)
# print("Hessian:")
# print(Hess)

# Step size 
gamma = 0.1/ np.max(np.abs(Hess))
print("gamma is:")
print(gamma)

# Train
x_star, costs = train_logistic_regression(A_train, b_train, lambd=1e-3, lr=gamma, max_iter=10000,tol=1e-8)

# # # Evaluate
# train_acc = accuracy(x_star, A_train, b_train)
# test_acc = accuracy(x_star, A_test, b_test)
# print(f"Train Accuracy: {train_acc * 100:.2f}%")
# print(f"Test Accuracy:  {test_acc * 100:.2f}%")

# print("True state:")
# print(x_star)

# # # Plot loss curve
# plt.plot(costs)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.grid(True)
# plt.show()

#############################################################################################################################


###################################################### Algorithm ############################################################
## Parameters for distributed 
N = 56 # Agent number
comDelay = 0
upDelay = 1
lb = -1
ub = 1

# Neighbor sets
neigh = np.tile(np.arange(0, N ).reshape(-1, 1), (1, N))

## Initial state
x0 = np.ones((1, 784, N))
y0 = np.ones((1, 784, N))

# ## Hyperparameters
# gammaHB = gammaNAG = gammaGM = gamma
# mu = 1e-3
# lambdaHB = (gammaNAG*mu)/2
# lambdaNAG = (gammaNAG*mu)/(2*(1-gammaNAG*mu))
# lambdaGM =(gammaGM*mu)/(2*(1-gammaGM*mu)); 
# betaGM =  (gammaGM*mu)/(2*(1-gammaGM*mu)); 

dt = 0.1
tf = 100
tvec = np.arange(0, tf + dt, dt)

num_samples, d = A_train.shape        # Total samples and feature dimension

# Split the features equally across agents
features_per_agent = d // N

A_train_list = []

start_idx = 0
for i in range(N):
    end_idx = start_idx + features_per_agent

    A_i = A_train[:, start_idx:end_idx]  # shape: (n, num_features)
    A_train_list.append(A_i)

    start_idx = end_idx

# initialization
trueStateGD = minusMinGD = np.zeros((len(tvec)+1, d))
xvecGD = np.zeros((len(tvec)+1, 784, N))

# probs = [1, 0.5, 0.1]
probs = [1]

if upDelay == 1:
    for prob in probs:
        probCom = prob * np.ones(N)

        opsk = 0
        xvecGD[0,:,:] = x0.copy()
        yvecGD = y0.copy()
        xvecNAG = x0.copy()
        yvecNAG = y0.copy()
        xvecHB = x0.copy()
        yvecHB = y0.copy()
        xvecGM = x0.copy()
        yvecGM = y0.copy()

        for k in range(len(tvec)):
            upif = np.where(np.random.rand(N) < probCom)[0] if upDelay == 1 else np.arange(N)

            for n in range(N):
                xvecGD[k+1,:,n] = xvecGD[k,:,n] # copy over
                # xvecHB[k+1,:,n] = xvecHB[k,:,n]
                # yvecHB[k+1,:,n] = yvecHB[k,:,n]
                # xvecNAG[k+1,:,n] = xvecNAG[k,:,n]
                # yvecNAG[k+1,:,n] = yvecNAG[k,:,n]
                # xvecGM[k+1,:,n] = xvecGM[k,:,n]
                # yvecGM[k+1,:,n] = yvecGM[k,:,n]

            if len(upif) > 0:
                for cur_agent in upif:
                    # cur_agent is the current agent index
                    # GD
                    x_cur = xvecGD[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (currently 14 features for every agent)
                    A_i = A_train_list[cur_agent]       # shape: (n, di)
                    # print(((b_train.reshape(-1,1) * (A_train @ x_cur))).shape)
                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ x_cur))))) + lambd * x_block
                    # print(f"Gradient for agent {cur_agent} is:")
                    # print(gradif)
                    xnew_block = projection(x_block - gamma * gradif.reshape(-1,1),lb,ub) # projected gradient descent
                    xvecGD[k+1,:,cur_agent] = x_cur[:,0] # copy over first
                    xvecGD[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = xnew_block[:,0] # update corresponding block for updating agent

                    # # NAG
                    # x_cur = xvecNAG[-1,:,cur_agent]
                    # ycur = yvecNAG[-1,:,cur_agent]
                    # zinp = x_cur + lambdaNAG * (x_cur - ycur)
                    # gradif =
                    # ynew = projection(x_cur[cur_agent] + lambdaNAG * (x_cur[cur_agent] - ycur[cur_agent]) - gammaNAG * gradif,lb,ub)
                    # yvecNAG[k+1,:,cur_agent] = ycur
                    # yvecNAG[k+1,cur_agent,cur_agent] = ynew
                    # ycur = yvecNAG[k+1,:,cur_agent]
                    # zinp = ycur + lambdaNAG * (ycur - x_cur)
                    # gradif = 2 * 
                    # xnew = np.clip(ycur[cur_agent] + lambdaNAG * (ycur[cur_agent] - x_cur[cur_agent]) - gammaNAG * gradif, lb,ub)
                    # xvecNAG[k+1,:,cur_agent] = x_cur
                    # xvecNAG[k+1,cur_agent,cur_agent] = xnew

                    # # HB
                    # x_cur = xvecHB[-1,:,cur_agent]
                    # ycur = yvecHB[-1,:,cur_agent]
                    # zinp = x_cur
                    # gradif = 
                    # ynew = np.clip(x_cur[cur_agent] + lambdaHB * (x_cur[cur_agent] - ycur[cur_agent]) - gammaHB * gradif, lb,ub)
                    # yvecHB[k+1,:,cur_agent] = ycur
                    # yvecHB[k+1,cur_agent,cur_agent] = ynew
                    # ycur = yvecHB[k+1,:,cur_agent]
                    # zinp = ycur
                    # gradif = 
                    # xnew = np.clip(ycur[cur_agent] + lambdaHB * (ycur[cur_agent] - x_cur[cur_agent]) - gammaHB * gradif, lb,ub)
                    # xvecHB[k+1,:,cur_agent] = x_cur
                    # xvecHB[k+1,cur_agent,cur_agent] = xnew

                    # # GM
                    # x_cur = xvecGM[-1,:,cur_agent]
                    # ycur = yvecGM[-1,:,cur_agent]
                    # zinp = x_cur + lambdaGM * (x_cur - ycur)
                    # gradif = 
                    # ynew = np.clip(x_cur[cur_agent] + betaGM * (x_cur[cur_agent] - ycur[cur_agent]) - gammaGM * gradif, lb,ub)
                    # yvecGM[k+1,:,cur_agent] = ycur
                    # yvecGM[k+1,cur_agent,cur_agent] = ynew
                    # ycur = yvecGM[k+1,:,cur_agent]
                    # zinp = ycur + lambdaGM * (ycur - x_cur)
                    # gradif = 
                    # xnew = np.clip(ycur[cur_agent] + betaGM * (ycur[cur_agent] - x_cur[cur_agent]) - gammaGM * gradif, lb,ub)
                    # xvecGM[k+1,:,cur_agent] = x_cur
                    # xvecGM[k+1,cur_agent,cur_agent] = xnew

            comif = np.where(np.random.rand(N) < probCom)[0] if comDelay == 1 else np.arange(N)

            if len(comif) > 0:
                for cur_agent in comif:
                    sendto = np.intersect1d(np.arange(N), neigh[:,cur_agent])
                    sendto = sendto[sendto != cur_agent]
                    for ii in sendto:
                        # Send to neighbors
                        xvecGD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = xvecGD[k+1,cur_agent*features_per_agent:(cur_agent + 1) *features_per_agent, cur_agent] # send to neighbors
                        # xvecNAG[k+1,cur_agent,ii] = xvecNAG[-1,cur_agent,cur_agent]
                        # yvecNAG[k+1,cur_agent,ii] = yvecNAG[-1,cur_agent,cur_agent]
                        # xvecHB[k+1,cur_agent,ii] = xvecHB[-1,cur_agent,cur_agent]
                        # yvecHB[k+1,cur_agent,ii] = yvecHB[-1,cur_agent,cur_agent]
                        # xvecGM[k+1,cur_agent,ii] = xvecGM[-1,cur_agent,cur_agent]
                        # yvecGM[k+1,cur_agent,ii] = yvecGM[-1,cur_agent,cur_agent]
            else:
                for n in range(N):
                    xvecGD[k+1,:,n] = xvecGD[k,:,n] # copy over
                    # xvecNAG[k+1,:,n] = xvecNAG[k,:,n]
                    # yvecNAG[k+1,:,n] = yvecNAG[k,:,n]
                    # xvecHB[k+1,:,n] = xvecHB[k,:,n]
                    # yvecHB[k+1,:,n] = yvecHB[k,:,n]
                    # xvecGM[k+1,:,n] = xvecGM[k,:,n]
                    # yvecGM[k+1,:,n] = yvecGM[k,:,n]

            print("Iteration:")
            print(k)

        # Loop through all features
        for i in range(d):
            trueStateGD[:,i] = xvecGD[:,i,i//features_per_agent]
            minusMinGD[:,i] = np.abs(trueStateGD[:,i] - x_star[i]*np.ones_like(trueStateGD[:,i]))

            # trueStateNAG[:,i] = xvecNAG[:,i,i]
            # minusMinNAG[:,i] = np.abs(trueStateNAG[:,i] - x_star[-1,0])

            # trueStateHB[:,i] = xvecHB[:,i,i]
            # minusMinHB[:,i] = np.abs(trueStateHB[:,i] - x_star[-1,0])

            # trueStateGAG[:,i] = xvecGM[:,i,i]
            # minusMinGAG[:,i] = np.abs(trueStateGAG[:,i] - x_star[-1,0])

        print("True state (GD):")
        print(x_star)
        print("True state (TAGD):")
        print(trueStateGD[-1,:])

        worstPerfGD = np.max(minusMinGD, axis=1)
        # worstPerfNAG = np.max(minusMinNAG, axis=1)
        # worstPerfHB = np.max(minusMinHB, axis=1)
        # worstPerfGAG = np.max(minusMinGAG, axis=1)

##############################################################################################################################


# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(tvec)+1), worstPerfGD, label='Worst-case GD', linewidth=2)
plt.xlabel('Time step')
plt.ylabel('Max Error ($\Vert x_i - x^* \Vert$)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()