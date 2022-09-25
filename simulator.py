import numpy as np

turns_per_timestep = 1
kill_threshold=0.3
predator_offspring_food_mutliplier = 1

class Simulator:
    def __init__(self, n_predators, n_prey):
        self.predators = self.generate_predators(n_predators)
        self.prey = self.generate_prey(n_prey)
    
    def generate_predators(self, n_predators):
        pass
    
    def generate_prey(self, n_prey):
        pass
        
    def test_pairs(self):
        order = np.random.shuffle(range(len(self.predators)))
        for i in order:
            curr_predator = self.predators[i]
            curr_prey = np.random.choice(self.prey)
            
            curr_eg, real = curr_prey.generate()
            pred_real = curr_predator.discriminate(curr_eg)
            
            # predator got it right
            if abs(pred_real-real) < kill_threshold:
                curr_predator.add_food(1-abs(real- pred_real))
                self.prey.remove(curr_prey)
        
    def timestep(self):
        for predator in self.predators:
            if predator.food == 0:
                self.predators.remove(predator)
                continue
            no_offspring = int(turns_per_timestep * predator_offspring_food_mutliplier * predator.food)
            if no_offspring > 0:
                pass