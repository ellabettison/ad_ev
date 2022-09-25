import torch.nn as nn
import torch
import numpy as np

class Predator:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.random.normal(loc=0, scale=0.1, size=2 * 128 + 128 * 64 + 64 * 1)
        else:
            self.genes = genes
        self.model = self.generate_network()

    def generate_network(self):
        model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        with torch.no_grad():
            self.model[0].weights = nn.Parameter(self.genes[:len(self.model[0].weights)])
            self.model[1].weights = nn.Parameter(self.genes[len(self.model[0].weights):len(self.model[0].weights)+len(self.model[1].weights)])
            self.model[2].weights = nn.Parameter(self.genes[-len(self.model[2].weights):])
        
        return model

    def generate(self):
        noise_inp = np.random.randn(2)
        return self.model(noise_inp)