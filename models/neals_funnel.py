
import torch
def funnel_ll(w):
    v_dist = torch.distributions.Normal(0,3)
    ll = v_dist.log_prob(w[0])
    x_dist = torch.distributions.Normal(0,torch.exp(-w[0])**0.5)
    ll += x_dist.log_prob(w[1:]).sum()
    return ll