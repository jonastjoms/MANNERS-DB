import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .bayesian_linear import BayesianLinear

class BayesianNetwork(nn.Module):

    def __init__(self, args):
        super(BayesianNetwork, self).__init__()
        # Maybe take in some information about data structure
        d, size = args.input_size
        self.task_outputs = args.task_outputs
        self.MC_samples = args.MC_samples
        self.device = args.device
        self.batch_size = args.batch_size
        self.init_lr = args.lr
        hidden_size = args.hidden_size

        # Two hidden layers:
        self.l1 = BayesianLinear(d*size, hidden_size, args)

        # Last layer
        self.classifier = nn.ModuleList()
        for task, output_size in self.task_outputs:
            self.classifier.append(BayesianLinear(hidden_size, output_size, args))

    def forward(self, x, sample=False, sample_last_layer=False):
        x = x.view(x.size(0),-1)
        x = F.relu(self.l1(x, sample))
        y = []
        for task, output_size in self.task_outputs:
            y.append(self.classifier[task](x, sample_last_layer))
        return y
