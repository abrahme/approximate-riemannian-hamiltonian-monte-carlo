import torch
def generate_bayesian_linear_regression_data(n_samples: int, n_covariates: int):
    torch.manual_seed(123)
    covariates = torch.distributions.Normal().sample(sample_shape=(n_covariates + 1, 1))

    X_samples = torch.distributions.Normal().sample(sample_shape=(n_samples,n_covariates))
    X = torch.hstack([torch.ones(size=(n_samples,1)), X_samples])

    y = torch.round(torch.logit(X @ covariates + torch.distributions.Normal().sample(sample_shape=(n_samples,1))*.1))
    return X, y, covariates