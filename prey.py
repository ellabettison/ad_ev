import torch.nn as nn
import torch
import numpy as np


class Prey:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.random.normal(loc=0, scale=0.1, size=2 * 16 + 16 * 32 + 32 * 2)
        else:
            self.genes = genes
        self.model = self.generate_network()

    def generate_network(self):
        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        with torch.no_grad():
            self.model[0].weights = nn.Parameter(self.genes[:len(self.model[0].weights)])
            self.model[1].weights = nn.Parameter(
                self.genes[len(self.model[0].weights):len(self.model[0].weights) + len(self.model[1].weights)])
            self.model[2].weights = nn.Parameter(self.genes[-len(self.model[2].weights):])

        return model

    def generate(self):
        noise_inp = np.random.randn(2)
        return self.model(noise_inp)
