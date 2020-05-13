import math
import torch

class Gaussian(torch.nn.Module):

    def __init__(self, mu, rho, device):
        super(Gaussian, self).__init__()
        self.mu = mu.to(device)
        self.rho = rho.to(device)
        self.device = device
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho)).to(self.device)

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        # Reparametrization
        posterior_sample = (self.mu + self.sigma * epsilon).to(self.device)
        return posterior_sample

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaledMixtureGaussian(torch.nn.Module):

    def __init__(self, args):
        super(ScaledMixtureGaussian, self).__init__()
        self.sigma1 = args.sigma1
        self.sigma2 = args.sigma2
        # Oops not traidtional, geometrical PI!
        self.pi = args.pi
        self.device = args.device
        self.sig1 = torch.tensor([math.exp(-1. * self.sigma1)], dtype=torch.float32, device = self.device)
        self.sig2 = torch.tensor([math.exp(-1. * self.sigma2)], dtype=torch.float32, device = self.device)
        self.gaussian1 = torch.distributions.Normal(0, self.sig1)
        self.gaussian2 = torch.distributions.Normal(0, self.sig2)

    def log_prob(self, input):
        input = input.to(self.device)
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi *prob1 + (1-self.pi) * prob2)).sum()
