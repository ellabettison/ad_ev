import numpy as np
from sklearn.base import BaseEstimator

from prey import Prey
from predator import Predator
import matplotlib.pyplot as plt

turns_per_timestep = 1

no_prey_vals = []
no_predators_vals = []

prey_losses = []


def evaluate_prey(prey):
    loss = 0
    for _ in range(100):
        eg, _ = prey.generate()
        eg = eg.detach().numpy()
        loss += abs(eg[1] - np.sin(eg[0] * np.pi))
    return loss


def custom_loss(y_true, y_pred):
    return abs(y_pred[1] - np.sin(y_pred[0] * np.pi))


class Simulator(BaseEstimator):
    def __init__(self, no_prey=None, no_predators=None, crossover_rate=None, mutation_sd=None, mutation_rate=None,
                 prey_offspring_no=None, predator_offspring_food_multiplier=None, kill_threshold=None, max_prey=None,
                 max_predators=None):
        self.no_prey = no_prey
        self.no_predators = no_predators
        self.crossover_rate = crossover_rate
        self.mutation_sd = mutation_sd
        self.mutation_rate = mutation_rate
        self.prey_offspring_no = prey_offspring_no
        self.predator_offspring_food_multiplier = predator_offspring_food_multiplier
        self.kill_threshold = kill_threshold
        self.max_prey = max_prey
        self.max_predators = max_predators
        if no_prey is not None:
            self.prey = self.generate_prey(self.no_prey)
        else:
            self.prey = []
        if no_predators is not None:
            self.predators = self.generate_predators(self.no_predators)
        else:
            self.predators = []
            
        self.best_so_far = -999999

    def set_params(self, **params):
        print(f"Testing params: {params}")
        self.no_predators = params["no_predators"]
        self.no_prey = params["no_prey"]
        self.predators = self.generate_predators(self.no_predators)
        self.prey = self.generate_prey(self.no_prey)
        self.max_predators = params["max_predators"]
        self.max_prey = params["max_prey"]
        self.kill_threshold = params["kill_threshold"]
        self.predator_offspring_food_multiplier = params["predator_offspring_food_multiplier"]
        self.prey_offspring_no = params["prey_offspring_no"]
        self.mutation_rate = params["mutation_rate"]
        self.mutation_sd = params["mutation_sd"]
        self.crossover_rate = params["crossover_rate"]
        self.best_so_far = -999999
        return self

    def get_params(self, deep=False):
        params = {
            "kill_threshold": self.kill_threshold,
            "predator_offspring_food_multiplier": self.predator_offspring_food_multiplier,
            "prey_offspring_no": self.prey_offspring_no,
            "mutation_rate": self.mutation_rate,
            "mutation_sd": self.mutation_sd,
            "crossover_rate": self.crossover_rate,
            "no_predators": self.no_predators,
            "no_prey": self.no_prey,
            "max_predators": self.max_predators,
            "max_prey": self.max_prey,
        }
        return params

    def fit(self, x_train, y_train):
        self.run_simulation(100)
        return self

    def generate_predators(self, n_predators):
        new_predators = []
        for _ in range(n_predators):
            new_predators.append(Predator())
        return new_predators

    def generate_prey(self, n_prey):
        new_prey = []
        for _ in range(n_prey):
            new_prey.append(Prey())
        return new_prey

    def reproduce_predator(self, predator, no_offspring):
        new_offspring = []
        other_parent = np.random.choice(self.predators)
        offspring = Predator(genes=self.crossover_pair(predator, other_parent))
        for _ in range(no_offspring):
            new_offspring.append(Predator(genes=self.mutate_org(offspring)))
        return new_offspring

    def reproduce_prey(self, prey):
        new_offspring = []
        other_parent = np.random.choice(self.prey)
        offspring = Prey(genes=self.crossover_pair(prey, other_parent))
        new_offspring.append(Prey(genes=self.mutate_org(offspring)))
        return new_offspring

    def crossover_pair(self, org1, org2):
        new_genes = []
        for i in range(len(org1.genes)):
            new_genes.append(
                np.random.choice([org1.genes[i], org2.genes[i]], p=[1 - self.crossover_rate, self.crossover_rate]))

    def mutate_org(self, org):
        new_genes = []
        for i in range(len(org.genes)):
            if np.random.random() < self.mutation_rate:
                new_genes.append(org.genes[i] + org.genes[i] * np.random.normal(loc=0, scale=self.mutation_sd))
            else:
                new_genes.append(org.genes[i])

    def test_pairs(self):
        predator_order = list(range(len(self.predators)))
        prey_order = list(range(len(self.prey)))
        np.random.shuffle(prey_order)
        np.random.shuffle(predator_order)
        print(f"No prey: {len(self.prey)}")
        print(f"No predators: {len(self.predators)}")
        no_prey_vals.append(len(self.prey))
        no_predators_vals.append(len(self.predators))
        for i, no in enumerate(predator_order[:int(min(len(self.predators), int(len(self.prey) / 1.5)))]):
            curr_predator = self.predators[no]
            # print(i, prey_order[i], len(self.prey))
            curr_prey = self.prey[prey_order[i]]

            curr_eg, real = curr_prey.generate_or_sample()
            pred_real = curr_predator.discriminate(curr_eg)

            # predator got it right
            if abs(pred_real - real) < self.kill_threshold:
                # print(f"Predator {curr_predator.name} ate prey {curr_prey.name}")
                curr_predator.add_food(1 - abs(real - pred_real))
                curr_prey.to_remove = True
            # else:
            # print(f"Predator {curr_predator.name} did not manage to eat prey {curr_prey.name}")

            if len(self.prey) == 1:
                return self.prey[0]

    def score(self, y_true, y_pred):
        score = -np.mean([evaluate_prey(p) for p in self.prey[:min(100, len(self.prey))]]) / min(100, len(self.prey))
        print(f"Curr score: {score}")
        if score > self.best_so_far:
            self.best_so_far = score
        return self.best_so_far

    def timestep(self):
        new_predators = []
        for predator in self.predators:
            if predator.food == 0:
                self.predators.remove(predator)
                if len(self.predators) == 0:
                    return self.prey[0]
                continue

            no_offspring = int(turns_per_timestep * self.predator_offspring_food_multiplier * predator.food)
            predator.food = 0

            if no_offspring > 0 and len(self.predators) + len(new_predators) < self.max_predators:
                new_predators += self.reproduce_predator(predator, no_offspring)
        self.predators += new_predators

        new_prey = []
        for prey in self.prey:
            if prey.to_remove:
                self.prey.remove(prey)
                if len(self.prey) == 1:
                    return self.prey[0]
            elif len(self.prey) + len(new_prey) < self.max_prey and np.random.random() < self.prey_offspring_no:
                new_prey += self.reproduce_prey(prey)
        prey_losses.append(self.score([], []))
        print(f"New prey loss: {prey_losses[-1]}")
        self.prey += new_prey

    def run_simulation(self, n_timesteps):
        for i in range(n_timesteps):
            for j in range(turns_per_timestep):
                final_prey = self.test_pairs()
                if final_prey is not None:
                    return final_prey
            final_prey = self.timestep()
            if final_prey is not None:
                return final_prey


if __name__ == "__main__":
    # params = {
    #     "kill_threshold": 0.5,
    #     "predator_offspring_food_multiplier": 2,
    #     "prey_offspring_no": 0.5,
    #     "mutation_rate": 0.15,
    #     "mutation_sd": 0.005,
    #     "crossover_rate": 0.33,
    #     "no_predators": 50,
    #     "no_prey": 50,
    #     "max_predators": 200,
    #     "max_prey": 200,
    # }

    params = {'prey_offspring_no': 0.9, 'predator_offspring_food_multiplier': 2.3, 'no_prey': 30, 'no_predators': 150, 'mutation_sd': 0.2, 'mutation_rate': 0.05, 'max_prey': 200, 'max_predators': 600, 'kill_threshold': 0.5, 'crossover_rate': 0.4}

    simulator = Simulator(**params)
    final_prey = simulator.run_simulation(50)
    if final_prey is None:
        final_prey = simulator.prey[0]

    xs = []
    ys = []
    for _ in range(1000):
        eg, _ = final_prey.generate()
        eg = eg.detach().numpy()
        xs.append(eg[0])
        ys.append(eg[1])

    plt.scatter(xs, ys)
    plt.plot(np.arange(-1, 1.5, 0.001), np.sin(np.arange(-1, 1.5, 0.001) * np.pi))
    plt.show()

    plt.plot(no_prey_vals, label="Prey")
    plt.plot(no_predators_vals, label="Predators")
    plt.plot(prey_losses, label="Prey losses")
    plt.legend()
    plt.show()
