 # A method for creating surrogate approximation to riemannian hamiltonian monte carlo for computational speedups

 ## Project Outline

 We first need to establish a baseline by benchmarking traditional Monte Carlo on two canonical examples

 1. Bayesian Linear Regression
 2. Neal's Funnel 

 We will scale up the number of samples needed from the posterior, as well as training samples and compute the following for various algorithms (Hamiltonian Monte Carlo, Riemannian Hamiltonian Monte Carlo, Approximate (Riemannian) Hamiltonian Monte Carlo w/ Basic Neural Net Parametrization, Approximate (Riemannian) Hamiltonian Monte Carlo w/ Symplectic Neural ODE Parametrization:

 1. Number of ESS per second (warm-up / sampling)
 2. Posterior Predictive Accuracy 
 3. Total Inference Time
 4. Number of Gradient Evaluations
 5. Posterior Sampling Speed


 Furthermore, we will compare the behavior of each of the Approximate (Riemannian) Hamiltonian Monte Carlo  methods by evaluating each's 
 preservation of Hamiltonian dynamics. 
