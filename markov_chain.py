
import random

import numpy as np

class History_Assignment:
    def make_Life_History(self, values, states):
        life_states = [states["careerfield"], states["education"], states["life-changing"], states["events2"], states["events"], states["romance"]]
        transitionMatrix = np.array([[0, 0.1, 0.2, 0.2, 0.2, 0.3], [0.4, 0, 0.15, 0.15, 0.15, 0.15], [0.2, 0.3, 0, 0.2, 0.2, 0.1],
                                    [0.2, 0.2, 0.2, 0, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0, 0.2], [0.3, 0.4, 0.1, 0.1, 0.1, 0.0]])

        num_steps = 4
        number = random.randint(0,5)
        if number == 0:
            life = states["careerfield"]
        elif number == 1:
            life = states["education"]
        elif number == 2:
            life = states["life-changing"]
        elif number == 3:
            life = states["events2"]
        elif number == 4:
            life = states["events"]
        elif number == 5:
            life = states["romance"]

        life_sequence = [life]
        for i in range(num_steps):
            life_probabilities = transitionMatrix[life_states.index(life), :]
            life = np.random.choice(life_states, p=life_probabilities)
            life_sequence.append(life)
        life_history = ", ".join(life_sequence)

        return life_history
