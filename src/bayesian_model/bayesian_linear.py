import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import Gaussian, ScaledMixtureGaussian

class BayesianLinear(nn.Module):

    # One linear bayesian layer

    def __init__(self, in_features, out_features, args):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = args.device
        self.rho = args.rho

        # Weight parameters:
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        self.weight_rho = nn.Parameter(self.rho + torch.empty((out_features, in_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        self.weight = Gaussian(self.weight_mu, self.weight_rho, self.device)

        # Bias parameters:
        self.bias_mu = nn.Parameter(torch.empty((out_features),
                                  device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
        self.bias_rho = nn.Parameter(self.rho + nn.Parameter(torch.empty(out_features,
                                  device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, self.device)

        # Prior distributions:
        self.weight_prior = ScaledMixtureGaussian(args)
        self.bias_prior = ScaledMixtureGaussian(args)

        # Log prior and lof posterior:
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
