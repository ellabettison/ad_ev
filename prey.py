import torch.nn as nn
import torch
import numpy as np

layer_sizes = [2, 16, 32, 2]

consonants = "qwrtpsdfghjklzxcvbnm"
vowels = "eyuioa"

class Prey:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.random.normal(loc=0, scale=0.1, size=layer_sizes[0] * layer_sizes[1] +
                                                                 layer_sizes[1] * layer_sizes[2] +
                                                                 layer_sizes[2] * layer_sizes[3])
        else:
            self.genes = genes
        self.model = self.generate_network()
        self.to_remove = False
        self.name = np.random.choice([*consonants]) +np.random.choice([*vowels]) +np.random.choice([*consonants])

    def generate_network(self):
        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            # nn.Sigmoid()
        )
        with torch.no_grad():
            model[0].weight.data = nn.Parameter(
                torch.reshape(torch.tensor(self.genes[:layer_sizes[0] * layer_sizes[1]]), (layer_sizes
                                                                                           [1], layer_sizes[0])))
            model[2].weight.data = nn.Parameter(torch.reshape(torch.tensor(self.genes[
                                                                           layer_sizes[0] * layer_sizes[1]:layer_sizes[
                                                                                                               0] *
                                                                                                           layer_sizes[
                                                                                                               1] +
                                                                                                           layer_sizes[
                                                                                                               1] *
                                                                                                           layer_sizes[
                                                                                                               2]]),
                                                              (layer_sizes[2], layer_sizes[1])))
            model[4].weight.data = nn.Parameter(
                torch.reshape(torch.tensor(self.genes[-layer_sizes[2] * layer_sizes[3]:]),
                              (layer_sizes[3], layer_sizes[2])))

        return model

    def generate(self):
        noise_inp = torch.tensor(np.random.randn(2))
        return self.model(noise_inp), 1

    def generate_or_sample(self):
        real_or_fake = np.random.random()
        if real_or_fake < 0.5:
            return self.generate()
        return self.sample_distribution()

    def sample_distribution(self):
        rand_x = np.random.random()
        rand_y = np.sin(rand_x*np.pi)
        return torch.tensor([rand_x, rand_y]), 0
