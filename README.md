# Bayesian Optimization Active Learning
Bayesian Optimization using active learning to query black box functions with unknown constraints. Completed as part of ETH Zurich Probabilistic AI class. Credit to Prof. Andreas Krause and the Probabilistic AI teaching team for project design as well as some skeleton code.

## Key Files
* **solution.py**: Includes definition of Bayesian Optimizer class with functions to calculate next best point from acquisition, query function at next point, and build Gaussian Process model. Also includes pre-defined evaluation functions. 
* **utils.py**: Includes functions to validate inputs and check function domains. 
