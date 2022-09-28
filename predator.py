import torch.nn as nn
import torch
import numpy as np

layer_sizes = [2,64,32,1]

consonants = "qwrtpsdfghjklzxcvbnm"
vowels = "eyuioa"

class Predator:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = np.random.normal(loc=0, scale=0.1, size=layer_sizes[0] * layer_sizes[1] + 
                                                                 layer_sizes[1] * layer_sizes[2] + 
                                                                layer_sizes[2 ] * layer_sizes[3])
        else:
            self.genes = genes
        self.model = self.generate_network()
        self.food = 0
        self.name = np.random.choice([*consonants]) +np.random.choice([*vowels]) +np.random.choice([*consonants])

    def generate_network(self):
        model = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.Sigmoid(),
        )
        
        with torch.no_grad():
            torch.tensor(self.genes[:])
            
            model[0].weight.data = nn.Parameter(torch.reshape(torch.tensor(self.genes[:layer_sizes[0]*layer_sizes[1]]), (layer_sizes[1], layer_sizes[0])))
            model[3].weight.data = nn.Parameter(torch.reshape(torch.tensor(self.genes[layer_sizes[0]*layer_sizes[1]:layer_sizes[0]*layer_sizes[1]+layer_sizes[1]*layer_sizes[2]]),(layer_sizes[2], layer_sizes[1])))
            model[6].weight.data = nn.Parameter(torch.reshape(torch.tensor(self.genes[-layer_sizes[2]*layer_sizes[3]:]),(layer_sizes[3], layer_sizes[2])))
        
        return model

    def discriminate(self, inp):
        return self.model(inp)
    
    def add_food(self, amt):
        self.food += amt