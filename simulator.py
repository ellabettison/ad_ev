import numpy as np
from prey import Prey
from predator import Predator

turns_per_timestep = 1
kill_threshold = 0.3
predator_offspring_food_mutliplier = 1
prey_offspring_no = 1
mutation_rate = 0.2
mutation_sd = 0.2


class Simulator:
    def __init__(self, n_predators, n_prey):
        self.predators = self.generate_predators(n_predators)
        self.prey = self.generate_prey(n_prey)

    def generate_predators(self, n_predators):
        pass

    def generate_prey(self, n_prey):
        pass

    def reproduce_predator(self, predator, no_offspring):
        new_offspring = []
        other_parent = np.random.choice(self.prey)
        offspring = Predator(genes=self.crossover_pair(predator, other_parent))
        for _ in range(no_offspring):
            new_offspring.append(Predator(genes=self.mutate_org(offspring)))

    def reproduce_prey(self, prey):
        new_offspring = []
        other_parent = np.random.choice(self.prey)
        offspring = Prey(genes=self.crossover_pair(prey, other_parent))
        for _ in range(prey_offspring_no):
            new_offspring.append(Prey(genes=self.mutate_org(offspring)))

    def crossover_pair(self, org1, org2):
        new_genes = []
        for i in range(len(org1.genes)):
            new_genes.append(np.random.choice([org1.genes[i], org2.genes[i]]))

    def mutate_org(self, org):
        new_genes = []
        for i in range(len(org.genes)):
            if np.random.random < mutation_rate:
                new_genes.append(org.genes[i] + org.genes[i] * np.random.normal(loc=0, scale=mutation_sd))
            else:
                new_genes.append(org.genes[i])

    def test_pairs(self):
        order = np.random.shuffle(range(len(self.predators)))
        for i in order:
            curr_predator = self.predators[i]
            curr_prey = np.random.choice(self.prey)

            curr_eg, real = curr_prey.generate()
            pred_real = curr_predator.discriminate(curr_eg)

            # predator got it right
            if abs(pred_real - real) < kill_threshold:
                curr_predator.add_food(1 - abs(real - pred_real))
                self.prey.remove(curr_prey)

    def timestep(self):
        for predator in self.predators:
            if predator.food == 0:
                self.predators.remove(predator)
                continue
            no_offspring = int(turns_per_timestep * predator_offspring_food_mutliplier * predator.food)
            if no_offspring > 0:
                self.reproduce_predator(predator, no_offspring)

        for prey in self.prey:
            self.reproduce_prey(prey)
