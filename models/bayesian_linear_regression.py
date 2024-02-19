
import torch
import torch.nn as nn
class BayesianLinearRegression(nn.Module):
    def __init__(self, input_dim: int):
        super(BayesianLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, out_features=1, bias = True)
    def forward(self, x):
        return torch.logit(self.linear(x))
    


    

