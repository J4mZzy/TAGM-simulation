A Totally Asynchronous Generalized Momentum Algorithm for Convex Optimization Simulation 
============

This is the simulation for the "A Totally Asynchronous Generalized Momentum Algorithm for Convex Optimization" paper. The example data set used here is the Fashion-MNIST dataset. A multi-class logarithmic regression model is used for benchmarking.

Comparisons between algorithms with update and communication probabilities of 0.5, 0.1, and 0.05:
![Description](Alg_comparison.png)

### How to run the simulation 
Set the variable in line 22 `recompte_true_cost` to `True` for the first run to generate the `true_cost.npy` file, this is used to save the optimal cost for comparison. Set this to `False` once the cost is calculated, and if any problem setting changes, this needs to be set to `True` again for the optimal cost to be re-calculated.

