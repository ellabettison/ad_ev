import sklearn
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
from simulator import Simulator
import numpy as np
from simulator import custom_loss

if __name__ == "__main__":
    space = {
        "kill_threshold": [0.5],
        "predator_offspring_food_multiplier": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9],
        "prey_offspring_no": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7],
        "mutation_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        "mutation_sd": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "crossover_rate": [0.1, 0.2, 0.3, 0.33, 0.4, 0.5],
        "no_predators": [10, 20, 30, 40, 50, 75, 100, 150, 200],
        "no_prey": [10, 20, 30, 40, 50, 75, 100, 150, 200],
        "max_predators": [200, 400, 600],
        "max_prey": [200, 400, 600],
    }

    simulator = Simulator()
    # scorer = make_scorer(custom_loss, greater_is_better=False)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    search = RandomizedSearchCV(simulator, space, cv=cv, n_iter=20)
    result = search.fit(np.arange(1000), np.ones(1000))

    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
