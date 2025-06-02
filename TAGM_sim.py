import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter


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
def train_logistic_regression(A, b, lambd=1e-3, lr=1e-3, max_iter=5000, tol=1e-6, print_cost = False):
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

def log_format_func(x, pos):
    exponent = int(np.log10(x))
    return r'$10^{{{}}}$'.format(exponent)

############################################## Global variables and parameters ################################################
## Set to true if the parameters needs to be re-learned, else the old paramters/costs will be loaded from the np files
recompute_problem = False
###############################################################################################################################

############################################## Data set #######################################################################
## Parse data set (if another data set is used, simply change it here in this code block)
# Load MNIST
mnist = fetch_openml('mnist_784', version=1)
X_all = mnist['data'].to_numpy().astype(np.float32) / 255.0  # (70000, 784)
y_all = mnist['target'].astype(int)
###############################################################################################################################

# Binary classification: digits 0 and 1
mask = (y_all == 0) | (y_all == 1)
A = X_all[mask]                     # shape: (N, 784)
b = np.where(y_all[mask] == 0, -1, 1)  # shape: (N,)

# Train/Test split (e.g., 80/20)
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)
num_samples, d = A_train.shape        # Total samples and feature dimension

# Initialize optimization variable x
x = np.zeros(A.shape[1])  # shape: (784,)
# Hessian checker 
Hess = logistic_regression_hessian(x,A_train,b_train,1e-3)  # ToDo: change to A_train, b_train
lambd_bound = least_diagonally_dominant(Hess)
lambd = 1e-3 # Regularization parameter 
# print(lambd_bound)
# print("Hessian:")
# print(Hess)

# Step size 
gamma = 0.1/ np.max(np.abs(Hess))
print("gamma is:")
print(gamma)

if recompute_problem:
    # Train
    x_star, costs = train_logistic_regression(A_train, b_train, lambd=1e-3, lr=gamma, max_iter=5000,tol=1e-8)
    np.save("x_star.npy", x_star) # Save the parameters 
    np.save("costs.npy", np.array(costs))


x_star = np.load("x_star.npy")
costs = np.load("costs.npy")
cost_star = costs[-1] # optimal cost

# print("True state:")
# print(x_star)
# print("Optimal cost:")
# print(cost_star)

# # # Evaluate
# train_acc = accuracy(x_star, A_train, b_train)
# test_acc = accuracy(x_star, A_test, b_test)
# print(f"Train Accuracy: {train_acc * 100:.2f}%")
# print(f"Test Accuracy:  {test_acc * 100:.2f}%")


# # # Plot loss curve
# plt.plot(costs)
# plt.xlabel("Iteration")
# plt.ylabel("Cost")
# plt.grid(True)
# plt.show()

#############################################################################################################################


###################################################### Algorithm ############################################################
## Parameters for distributed implementation
N = 56 # agent number
comDelay = 0
upDelay = 1
lb = -1 # lower bound for projection 
ub = 1  # upper bound for projection 

# Neighbor sets
neigh = np.tile(np.arange(0, N).reshape(-1, 1), (1, N))

## Initial states 
x0 = np.ones((d, N)) # all 1's across all agents 
y0 = np.ones((d, N)) # all 1's across all agents 

## Hyperparameters
gamma = gamma * 0.1
gammaHB = gamma * 2
gammaNAG = gamma * 4
gammaGM = gamma * 6
mu = 1e-3  # same as the lambd for now
lambdaHB = (gammaNAG*mu)/2
lambdaNAG = (gammaNAG*mu)/(2*(1-gammaNAG*mu))
lambdaGM = (gammaGM*mu)/(2*(1-gammaGM*mu))*0.1; 
betaGM =  (gammaGM*mu)/(2*(1-gammaGM*mu)); 

## Display hyperparamters 
print("lambdaHB is:")
print(lambdaHB)
print("lambdaNAG is:")
print(lambdaNAG)
print("lambdaGM is:")
print(lambdaGM)
print("betaGM is:")
print(betaGM)

## Time vector (iterations)
dt = 0.1
tf = 100 
tvec = np.arange(0, tf + dt, dt)

# Split the features equally across agents
features_per_agent = d // N   # perferably divided evenly 

# Divide the feature space for each agent
A_train_list = []

start_idx = 0
for i in range(N):
    end_idx = start_idx + features_per_agent

    A_i = A_train[:, start_idx:end_idx]  # shape: (n, num_features)
    A_train_list.append(A_i)

    start_idx = end_idx

# Initializations
xvecGD = np.zeros((len(tvec)+1, 784, N))
xvecHB = np.zeros((len(tvec)+1, 784, N))
yvecHB = np.zeros((len(tvec)+1, 784, N))
xvecNAG = np.zeros((len(tvec)+1, 784, N))
yvecNAG = np.zeros((len(tvec)+1, 784, N))
xvecGM = np.zeros((len(tvec)+1, 784, N))
yvecGM = np.zeros((len(tvec)+1, 784, N))

trueStateGD = np.ones((len(tvec), d))
worstPerfGD = np.zeros(len(tvec))
trueStateHB = np.ones((len(tvec), d))
worstPerfHB = np.zeros(len(tvec))
trueStateNAG = np.ones((len(tvec), d))
worstPerfNAG = np.zeros(len(tvec))
trueStateGM = np.ones((len(tvec), d))
worstPerfGM = np.zeros(len(tvec))

avg_costs = []

# probabily array for updating 
# probs = [1, 0.5, 0.1]
probs = [1]

if upDelay == 1:
    for prob in probs:
        probCom = prob * np.ones(N)

        # initialize x0 and y0
        xvecGD[0,:,:] = x0.copy() 
        xvecHB[0,:,:] = x0.copy()
        yvecHB[0,:,:] = y0.copy()
        xvecNAG[0,:,:] = x0.copy()
        yvecNAG[0,:,:] = y0.copy()
        xvecGM[0,:,:] = x0.copy()
        yvecGM[0,:,:] = y0.copy()

        for k in range(len(tvec)):
            upif = np.where(np.random.rand(N) < probCom)[0] if upDelay == 1 else np.arange(N)

            for n in range(N):
                ## Copy over
                xvecGD[k+1,:,n] = xvecGD[k,:,n] 
                xvecHB[k+1,:,n] = xvecHB[k,:,n]
                yvecHB[k+1,:,n] = yvecHB[k,:,n]
                xvecNAG[k+1,:,n] = xvecNAG[k,:,n]
                yvecNAG[k+1,:,n] = yvecNAG[k,:,n]
                xvecGM[k+1,:,n] = xvecGM[k,:,n]
                yvecGM[k+1,:,n] = yvecGM[k,:,n]

            if len(upif) > 0:
                for cur_agent in upif:
                    # cur_agent is the current agent index

                    ####################################################### GD #############################################################################################
                    x_cur = xvecGD[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (currently 14 features for every agent)
                    A_i = A_train_list[cur_agent]       # shape: (n, di)
                    # print(((b_train.reshape(-1,1) * (A_train @ x_cur))).shape)
                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ x_cur))))) + lambd * x_block # gradient 
                    # print(f"Gradient for agent {cur_agent} is:")
                    # print(gradif)
                    xnew_block = projection(x_block - gamma * gradif.reshape(-1,1),lb,ub) # projected gradient descent
                    xvecGD[k+1,:,cur_agent] = x_cur[:,0] # copy over first
                    xvecGD[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = xnew_block[:,0] # update corresponding block for updating agent


                    ####################################################### HB #############################################################################################
                    x_cur = xvecHB[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (x)
                    y_cur = yvecHB[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)

                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ x_cur))))) + lambd * x_block # gradient 
                    ynew_block = projection(x_block - gammaHB * gradif.reshape(-1,1) + lambdaHB * (x_block-y_block) ,lb,ub) # projected gradient descent
                    yvecHB[k+1,:,cur_agent] = y_cur[:,0] # copy over first
                    yvecHB[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = ynew_block[:,0] # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecHB[k+1,:,cur_agent].reshape(-1,1) # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)
                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ y_cur))))) + lambd * y_block # gradient 
                    xnew_block = projection(y_block - gammaHB * gradif.reshape(-1,1) + lambdaHB * (y_block-x_block) ,lb,ub) # projected gradient descent
                    xvecHB[k+1,:,cur_agent] = x_cur[:,0] # copy over first
                    xvecHB[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = xnew_block[:,0] # update corresponding block for updating agent


                    ####################################################### NAG ############################################################################################
                    x_cur = xvecNAG[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (x)
                    y_cur = yvecNAG[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)
                    
                    # Momentum on x
                    x_momentum = x_cur + lambdaNAG*(x_cur-y_cur) # gradient is evaluated here
                    x_momentum_block = x_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block with momentum added (x)
                    
                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ x_momentum))))) + lambd * x_momentum_block # gradient 
                    ynew_block = projection(x_block - gammaNAG * gradif.reshape(-1,1) + lambdaNAG * (x_block-y_block) ,lb,ub) # projected gradient descent
                    yvecNAG[k+1,:,cur_agent] = y_cur[:,0] # copy over first
                    yvecNAG[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = ynew_block[:,0] # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecNAG[k+1,:,cur_agent].reshape(-1,1) # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)

                    # Momentum on y
                    y_momentum = y_cur + lambdaNAG*(y_cur-x_cur) # gradient is evaluated here
                    y_momentum_block = y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block with momentum added (x)

                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ y_momentum))))) + lambd * y_momentum_block # gradient 
                    xnew_block = projection(y_block - gammaNAG * gradif.reshape(-1,1) + lambdaNAG * (y_block-x_block) ,lb,ub) # projected gradient descent
                    xvecNAG[k+1,:,cur_agent] = x_cur[:,0] # copy over first
                    xvecNAG[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = xnew_block[:,0] # update corresponding block for updating agent

                    ####################################################### GM #############################################################################################
                    x_cur = xvecGM[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (x)
                    y_cur = yvecGM[k,:,cur_agent].reshape(-1,1) # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)
                    
                    # Momentum on x
                    x_momentum = x_cur + lambdaGM*(x_cur-y_cur) # gradient is evaluated here
                    x_momentum_block = x_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block with momentum added (x)
                    
                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ x_momentum))))) + lambd * x_momentum_block # gradient 
                    ynew_block = projection(x_block - gammaGM * gradif.reshape(-1,1) + betaGM * (x_block-y_block) ,lb,ub) # projected gradient descent
                    yvecGM[k+1,:,cur_agent] = y_cur[:,0] # copy over first
                    yvecGM[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = ynew_block[:,0] # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecGM[k+1,:,cur_agent].reshape(-1,1) # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block (y)

                    # Momentum on y
                    y_momentum = y_cur + lambdaGM*(y_cur-x_cur) # gradient is evaluated here
                    y_momentum_block = y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent].reshape(-1,1)  # current agent's local block with momentum added (x)

                    gradif = (1/num_samples) * (A_i.T @ (-b_train.reshape(-1,1)/ (1 + np.exp(b_train.reshape(-1,1) * (A_train @ y_momentum))))) + lambd * y_momentum_block # gradient 
                    xnew_block = projection(y_block - gammaGM * gradif.reshape(-1,1) + betaGM * (y_block-x_block) ,lb,ub) # projected gradient descent
                    xvecGM[k+1,:,cur_agent] = x_cur[:,0] # copy over first
                    xvecGM[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent, cur_agent] = xnew_block[:,0] # update corresponding block for updating agent

            comif = np.where(np.random.rand(N) < probCom)[0] if comDelay == 1 else np.arange(N) # whether or not we need to communicate

            if len(comif) > 0:
                for cur_agent in comif:
                    sendto = np.intersect1d(np.arange(N), neigh[:,cur_agent])
                    sendto = sendto[sendto != cur_agent]
                    for ii in sendto:
                        # Send to neighbors
                        xvecGD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = xvecGD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        xvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = xvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        yvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = yvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        xvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = xvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        yvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = yvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        xvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = xvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
                        yvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, ii] = yvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent, cur_agent] # send local true states to neighbors
            else:
                for n in range(N):
                    xvecGD[k+1,:,n] = xvecGD[k,:,n] # copy over
                    xvecHB[k+1,:,n] = xvecHB[k,:,n] # copy over
                    yvecHB[k+1,:,n] = yvecHB[k,:,n] # copy over
                    xvecNAG[k+1,:,n] = xvecNAG[k,:,n] # copy over
                    yvecNAG[k+1,:,n] = yvecNAG[k,:,n] # copy over
                    xvecGM[k+1,:,n] = xvecGM[k,:,n] # copy over
                    yvecGM[k+1,:,n] = yvecGM[k,:,n] # copy over
            
            for agent in range(N):
                idx_start = agent * features_per_agent
                idx_end = (agent + 1) * features_per_agent
                trueStateGD[k, idx_start:idx_end] = xvecGD[k, idx_start:idx_end, agent]
                trueStateHB[k, idx_start:idx_end] = xvecHB[k, idx_start:idx_end, agent]
                trueStateNAG[k, idx_start:idx_end] = xvecNAG[k, idx_start:idx_end, agent]
                trueStateGM[k, idx_start:idx_end] = xvecGM[k, idx_start:idx_end, agent]
        
                 
            worstPerfGD[k] = np.max(np.abs(trueStateGD[k, :] - x_star))
            worstPerfHB[k] = np.max(np.abs(trueStateHB[k, :] - x_star))
            worstPerfNAG[k] = np.max(np.abs(trueStateNAG[k, :] - x_star))
            worstPerfGM[k] = np.max(np.abs(trueStateGM[k, :] - x_star))

            print("Iteration:")
            print(k+1)

        # print("True state (GD):")
        # print(x_star)
        # print("True state (TAGD):")
        # print(trueStateGD[-1,:]) 

        # # # Evaluate
        # train_acc = accuracy(trueStateGD[-1,:], A_train, b_train)
        # test_acc = accuracy(trueStateGD[-1,:], A_test, b_test)
        # print(f"Train Accuracy: {train_acc * 100:.2f}%")
        # print(f"Test Accuracy:  {test_acc * 100:.2f}%")

##############################################################################################################################


# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Color theme based on pp
pp = 1  # Change to 2, 3, or 4 for other color schemes
colors = {1: 'b', 2: 'r', 3: 'g', 4: 'm'}
c = colors.get(pp, 'k')

plt.figure(figsize=(8, 5))

# Plot GD and HB
line1, = plt.semilogy(np.arange(len(tvec)), worstPerfGD, f'{c}-', linewidth=2, label='GD')
line2, = plt.semilogy(np.arange(len(tvec)), worstPerfHB, f'{c}--', linewidth=2, label='HB')
line3, = plt.semilogy(np.arange(len(tvec)), worstPerfNAG, f'{c}-.', linewidth=2, label='NAG')
line4, = plt.semilogy(np.arange(len(tvec)), worstPerfGM, f'{c}-o', linewidth=2, label='GM')

# # Optional: probability markers
# p1, = plt.semilogy([], [], 'b*', markersize=8, markerfacecolor='b', linewidth=2, label=f'{probs[0]:.1f}')
# p2, = plt.semilogy([], [], 'r*', markersize=8, markerfacecolor='r', linewidth=2, label=f'{probs[1]:.1f}')
# p3, = plt.semilogy([], [], 'g*', markersize=8, markerfacecolor='g', linewidth=2, label=f'{probs[2]:.1f}')

# Labels and formatting
plt.xlabel(r'Iterations, $k$', fontsize=14)
plt.ylabel(r'$\max\limits_{i\in\mathcal{V}} \|x^i - x^\star\|$', fontsize=14)
plt.xlim([0, k+10])
plt.grid(True, which='both')

# Log scale formatting
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'$10^{{{int(np.log10(val))}}}$' if val != 0 else '0'))

# Legends
legend1 = plt.legend([line1, line2, line3, line4], ['GD', 'HB', 'NAG', 'GM'], loc='upper right', fontsize=12, title='Algorithm')
# legend2 = plt.legend([p1, p2, p3], [f'{probs[0]:.1f}', f'{probs[1]:.1f}', f'{probs[2]:.1f}'],
#                      loc='upper right', fontsize=12, title='Probability')

# Add the main legend back
plt.gca().add_artist(legend1)
plt.tight_layout()
plt.savefig("Alg_comparison.png", dpi=300)
plt.show()

# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(len(tvec)+1), avgCostDiffGD, label='Avg Cost - Optimal Cost', linewidth=2)
# plt.xlabel('Time step')
# plt.ylabel('Average Cost Error')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()