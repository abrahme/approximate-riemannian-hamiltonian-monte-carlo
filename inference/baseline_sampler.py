

import torch
import hamiltorch
from hamiltorch.samplers import Sampler
from ..models.neals_funnel import funnel_ll
from ..models.bayesian_linear_regression import BayesianLinearRegression
from ..data.synthetic_data import generate_bayesian_linear_regression_data


softabs_const=10**6


def sample_neal(D: int, L: int, step_size:float, num_samples:int, sampler = None):
    hamiltorch.set_random_seed(123)
    params_init = torch.ones(D + 1)
    params_init[0] = 0.

    params_hmc = hamiltorch.sample(log_prob_func=funnel_ll, params_init=params_init, num_samples=num_samples,
                               step_size=step_size, num_steps_per_sample=L, sampler = Sampler.RMHMC if sampler else Sampler.HMC,
                               softabs_const=softabs_const if sampler else None, 
                               metric= hamiltorch.Metric.SOFTABS if sampler else hamiltorch.Metric.HESSIAN,
                               )
    
    return params_hmc 


def sample_bayesian_logistic_regression(n_covariates:int, n_samples: int, n_posterior_samples:int, step_size:float, L:int,
                                        sampler = None):
    hamiltorch.set_random_seed(123)
    params_init = torch.ones(n_covariates + 1)
    X, y = generate_bayesian_linear_regression_data(n_samples=n_samples, n_covariates=n_covariates)
    
    params_hmc = hamiltorch.sample_model(BayesianLinearRegression(n_covariates), X, y, params_init=params_init,
                            model_loss = "binary_class_linear_output", 
                            step_size=step_size, num_samples=n_posterior_samples, num_steps_per_sample=L, 
                            sampler = Sampler.RMHMC if sampler else Sampler.HMC,
                            softabs_const=softabs_const if sampler else None, 
                            metric= hamiltorch.Metric.SOFTABS if sampler else hamiltorch.Metric.HESSIAN
                            )
    
    return params_hmc

