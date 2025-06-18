import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Global variable for recomputing true cost 
recompte_true_cost = False

######################################################### Data set ###################################################
## Fashion-MNIST
fmnist = fetch_openml(name="Fashion-MNIST", version=1)
A_all = fmnist['data'].to_numpy().astype(np.float32) / 255.0  # (70000, 784)
y_all = fmnist['target'].astype(int).values.reshape(-1,1)

# Shuffle and train/test split
A_train, A_test, y_train, y_test = train_test_split(A_all, y_all, test_size=0.3, random_state=77)

# Standardize the data
scaler = StandardScaler()
A_train = scaler.fit_transform(A_train)
A_test = scaler.transform(A_test)

# PCA (optional)
# pca = PCA(n_components=700, whiten=True)
# A_train = pca.fit_transform(A_train)
# X_test = pca.transform(X_test)

# Reshape labels
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Total samples and feature dimension
num_samples, d = A_train.shape        

# Hot encoder 
onehot_encoder = OneHotEncoder(sparse=False)

####################################################### Function block #################################################
# Define the projection argument for projected gradient descent
def projection(arr, lower_bound, upper_bound):
    return np.clip(np.asarray(arr), lower_bound, upper_bound)

def loss(X, Y, W, mu):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1)))) + mu/2 * np.linalg.norm(W, ord='fro') ** 2
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + mu * W
    return gd

## Y is the labels, X is the data
def gradient_descent(X, Y, max_iter=2000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
 
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W, mu))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W, loss_lst[-1]

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W, self.true_cost = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
    
#######################################################################################################################

################################################# True cost computation ###############################################
if recompte_true_cost == True:
    import time

    start_time = time.time()

    # Fit model
    model = Multiclass()
    model.fit(A_train, y_train)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"Elapsed time: {minutes} minutes and {seconds:.2f} seconds")

    # Plot loss
    model.loss_plot()

    # Percentage of True values 
    accuracy = np.mean(model.predict(A_test) == y_test.ravel()) * 100

    print(f"Accuracy: {accuracy:.2f}%")

    ## Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Print Confusion Matrix
    conf_matrix =  confusion_matrix(y_test, model.predict(A_test))
    # Plot it
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(range(10)))
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Normalized Confusion Matrix")
    plt.show()

########################################################################################################################

################################################### Algorithm block ####################################################
## Parameters for distributed implementation
N = 56 # agent number
comDelay = 1
upDelay = 1
lb = -5 # lower bound for projection 
ub = 5  # upper bound for projection 
mu = 1e-2

# Neighbor sets
neigh = np.tile(np.arange(0, N).reshape(-1, 1), (1, N))

K = 10 # number of classes

## Initial states 
x0 = np.zeros((d, K, N), dtype=np.float32) # all 0's across all agents 
y0 = np.zeros((d, K, N), dtype=np.float32) # all 0's across all agents 

## Hyperparameters
gamma = 1e-2
gammaHB = gamma   
gammaNAG = gamma   
gammaGM = gamma 

lambdaHB = (gammaHB*20)/2*0.5
lambdaNAG = (gammaNAG*20)/(2*(1-gammaNAG*20))*0.7
lambdaGM = (gammaGM*20)/(2*(1-gammaGM*20))*0.1 
betaGM =  (gammaGM*20)/(2*(1-gammaGM*20)) 

# ## Display hyperparamters 
# print("lambdaHB is:")
# print(lambdaHB)
# print("lambdaNAG is:")
# print(lambdaNAG)
# print("lambdaGM is:")
# print(lambdaGM)
# print("betaGM is:")
# print(betaGM)

# Time vector (iterations)
dt = 0.1
tf = 800 
tvec = np.arange(0, tf + dt, dt)

# Split the features equally across agents
features_per_agent = d // N   # perferably divided evenly 

# Divide the feature space for each agent
b_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))

# A_train = X
# Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
# b_train = Y_onehot

A_train_list = []

start_idx = 0
for i in range(N):
    end_idx = start_idx + features_per_agent

    A_i = A_train[:, start_idx:end_idx]  # shape: (n, num_features)
    A_train_list.append(A_i)

    start_idx = end_idx

# Initializations
xvecGD = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
xvecHB = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
yvecHB = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
xvecNAG = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
yvecNAG = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
xvecGM = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)
yvecGM = np.zeros((len(tvec)+1, d, K, N), dtype=np.float32)


trueStateGD = np.ones((len(tvec), d, K), dtype=np.float32)
trueStateHB = np.ones((len(tvec), d, K), dtype=np.float32)
trueStateNAG = np.ones((len(tvec), d, K), dtype=np.float32)
trueStateGM = np.ones((len(tvec), d, K), dtype=np.float32)

# cost lists
cost_diffGD_lst = []
cost_diffHB_lst = []
cost_diffNAG_lst = []
cost_diffGM_lst = []

true_cost = np.load("true_cost.npy").astype(np.float32)
probs = [0.1]

if upDelay == 1:
    for prob in probs:
        probCom = prob * np.ones(N)

        # initialize x0 and y0
        xvecGD[0,:,:,:] = x0.copy() 
        xvecHB[0,:,:,:] = x0.copy()
        yvecHB[0,:,:,:] = y0.copy()
        xvecNAG[0,:,:,:] = x0.copy()
        yvecNAG[0,:,:,:] = y0.copy()
        xvecGM[0,:,:,:] = x0.copy()
        yvecGM[0,:,:,:] = y0.copy()

        for k in range(len(tvec)):
            upif = np.where(np.random.rand(N) < probCom)[0] if upDelay == 1 else np.arange(N)

            for n in range(N):
                ## Copy over
                xvecGD[k+1,:,:,n] = xvecGD[k,:,:,n] # copy over
                xvecHB[k+1,:,:,n] = xvecHB[k,:,:,n] # copy over
                yvecHB[k+1,:,:,n] = yvecHB[k,:,:,n] # copy over
                xvecNAG[k+1,:,:,n] = xvecNAG[k,:,:,n] # copy over
                yvecNAG[k+1,:,:,n] = yvecNAG[k,:,:,n] # copy over
                xvecGM[k+1,:,:,n] = xvecGM[k,:,:,n] # copy over
                yvecGM[k+1,:,:,n] = yvecGM[k,:,:,n] # copy over

            if len(upif) > 0:
                for cur_agent in upif:
                    # cur_agent is the current agent index

                    ####################################################### GD #############################################################################################
                    x_cur = xvecGD[k,:,:,cur_agent] # current agent's entire state vetor
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  
                    A_i = A_train_list[cur_agent]       # shape: (n, di)
                    # print(((b_train.reshape(-1,1) * (A_train @ x_cur))).shape)

                    Z = - A_train @ x_cur
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_block
                    
                    xnew_block = projection(x_block - gamma * gradif,lb,ub) # projected gradient descent
                    xvecGD[k+1,:,:,cur_agent] = x_cur # copy over first
                    xvecGD[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = xnew_block # update corresponding block for updating agent

                    ####################################################### HB #############################################################################################
                    x_cur = xvecHB[k,:,:,cur_agent] # current agent's entire state vetor (x)
                    y_cur = yvecHB[k,:,:,cur_agent] # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)
    
                    Z = - A_train @ x_cur
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_block
                    
                    ynew_block = projection(x_block-gammaHB*gradif+lambdaHB*(x_block-y_block),lb,ub) # projected gradient descent with momentum
                    yvecHB[k+1,:,:,cur_agent] = y_cur # copy over first
                    yvecHB[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = ynew_block # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecHB[k+1,:,:,cur_agent] # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)

                    Z = - A_train @ y_cur
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_block
                    
                    xnew_block = projection(y_block-gammaHB*gradif+lambdaHB*(y_block-x_block),lb,ub) # projected gradient descent with momentum
                    xvecHB[k+1,:,:,cur_agent] = x_cur # copy over first
                    xvecHB[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = xnew_block # update corresponding block for updating agent

                    ####################################################### NAG ############################################################################################
                    x_cur = xvecNAG[k,:,:,cur_agent] # current agent's entire state vetor (x)
                    y_cur = yvecNAG[k,:,:,cur_agent] # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)
                    
                    # Momentum on x
                    x_momentum = x_cur + lambdaNAG*(x_cur-y_cur) # gradient is evaluated here
                    
                    Z = - A_train @ x_momentum
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * x_momentum[cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:]

                    ynew_block = projection(x_block-gammaNAG*gradif+lambdaNAG*(x_block-y_block),lb,ub) # projected gradient descent with momentum
                    yvecNAG[k+1,:,:,cur_agent] = y_cur # copy over first
                    yvecNAG[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = ynew_block # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecNAG[k+1,:,:,cur_agent] # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)

                    # Momentum on y
                    y_momentum = y_cur + lambdaNAG*(y_cur-x_cur) # gradient is evaluated here

                    Z = - A_train @ y_momentum
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]
                    
                    xnew_block = projection(y_block-gammaNAG*gradif+lambdaNAG*(y_block-x_block),lb,ub) # projected gradient descent with momentum
                    xvecNAG[k+1,:,:,cur_agent] = x_cur # copy over first
                    xvecNAG[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = xnew_block # update corresponding block for updating agent

                    ####################################################### GM #############################################################################################
                    x_cur = xvecGM[k,:,:,cur_agent] # current agent's entire state vetor (x)
                    y_cur = yvecGM[k,:,:,cur_agent] # current agent's entire state vetor (y)
                    ## Descent on y first 
                    x_block = x_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (x)
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)
                    
                    # Momentum on x
                    x_momentum = x_cur + lambdaGM*(x_cur-y_cur) 
                    
                    Z = - A_train @ x_momentum
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu *  x_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:] 
                    
                    ynew_block = projection(x_block-gammaGM*gradif+betaGM*(x_block-y_block),lb,ub) # projected gradient descent with momentum
                    yvecGM[k+1,:,:,cur_agent] = y_cur # copy over first
                    yvecGM[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = ynew_block # update corresponding block for updating agent

                    ## Descent on x
                    y_cur = yvecGM[k+1,:,:,cur_agent] # the current y is updated
                    y_block = y_cur[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]  # current agent's local block (y)

                    # Momentum on y
                    y_momentum = y_cur + lambdaGM*(y_cur-x_cur) 

                    Z = - A_train @ y_momentum
                    P = softmax(Z, axis=1)
                    gradif = 1/num_samples * (A_i.T @ (b_train - P)) + mu * y_momentum[cur_agent * features_per_agent : (cur_agent + 1) * features_per_agent,:]
                    
                    xnew_block = projection(y_block-gammaGM*gradif+betaGM*(y_block-x_block),lb,ub) # projected gradient descent with momentum
                    xvecGM[k+1,:,:,cur_agent] = x_cur # copy over first
                    xvecGM[k+1, cur_agent*features_per_agent :(cur_agent + 1)*features_per_agent,:,cur_agent] = xnew_block # update corresponding block for updating agent

            comif = np.where(np.random.rand(N) < probCom)[0] if comDelay == 1 else np.arange(N) # whether or not we need to communicate

            if len(comif) > 0:
                for cur_agent in comif:
                    sendto = np.intersect1d(np.arange(N), neigh[:,cur_agent])
                    sendto = sendto[sendto != cur_agent]
                    for ii in sendto:
                        # Send to neighbors
                        xvecGD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = xvecGD[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        xvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = xvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        yvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = yvecHB[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        xvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = xvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        yvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = yvecNAG[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        xvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = xvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
                        yvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, ii] = yvecGM[k+1, cur_agent*features_per_agent:(cur_agent + 1)*features_per_agent,:, cur_agent] # send local true states to neighbors
            else:
                for n in range(N):
                    xvecGD[k+1,:,:,n] = xvecGD[k,:,:,n] # copy over
                    xvecHB[k+1,:,:,n] = xvecHB[k,:,:,n] # copy over
                    yvecHB[k+1,:,:,n] = yvecHB[k,:,:,n] # copy over
                    xvecNAG[k+1,:,:,n] = xvecNAG[k,:,:,n] # copy over
                    yvecNAG[k+1,:,:,n] = yvecNAG[k,:,:,n] # copy over
                    xvecGM[k+1,:,:,n] = xvecGM[k,:,:,n] # copy over
                    yvecGM[k+1,:,:,n] = yvecGM[k,:,:,n] # copy over
            for agent in range(N):
                idx_start = agent * features_per_agent
                idx_end = (agent + 1) * features_per_agent
                
                trueStateGD[k, idx_start:idx_end,:] = xvecGD[k, idx_start:idx_end,:,agent]
                trueStateHB[k, idx_start:idx_end,:] = xvecHB[k, idx_start:idx_end,:,agent]
                trueStateNAG[k, idx_start:idx_end,:] = xvecNAG[k, idx_start:idx_end,:,agent]
                trueStateGM[k, idx_start:idx_end,:] = xvecGM[k, idx_start:idx_end,:,agent]
                
            ## true state is used to caluclate the true cost
            # can also load the saved true cost
            cost_diffGD_lst.append(loss(A_train, b_train, trueStateGD[k, :,:], mu) - true_cost)
            cost_diffHB_lst.append(loss(A_train, b_train, trueStateHB[k, :,:], mu) - true_cost)
            cost_diffNAG_lst.append(loss(A_train, b_train, trueStateNAG[k, :,:], mu) - true_cost)
            cost_diffGM_lst.append(loss(A_train, b_train, trueStateGM[k, :,:], mu) - true_cost)

            print("Iteration:")
            print(k+1)

        # print("True state (GD):")
        # print(x_star)
        # print("True state (TAGD):")
        # print(trueStateGD[-1,:]) 
        # print(f"True cost is: {cost_diffGD_lst[-1]}")

## Save data
# can be loaded in another script for plotting
for prob in probs:
    prob_pct = int(prob * 100) 
    np.save(f"cost_diff_GD_prob={prob_pct}.npy", np.array(cost_diffGD_lst))
    np.save(f"cost_diff_HB_prob={prob_pct}.npy", np.array(cost_diffHB_lst))
    np.save(f"cost_diff_NAG_prob={prob_pct}.npy", np.array(cost_diffNAG_lst))
    np.save(f"cost_diff_GM_prob={prob_pct}.npy", np.array(cost_diffGM_lst))

##############################################################################################################################
