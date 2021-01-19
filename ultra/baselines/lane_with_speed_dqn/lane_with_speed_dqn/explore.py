import numpy as np
from torch import nn


class EpsilonExplore:
    """Controls a value epsilon by decreasing its value every time
    step to increase exploitation over exploration as time goes on.
    Initially, the epsilon value is set to the max_epsilon value. If
    decay is specified as an integer, epsilon is decreased by
    epsilon_step * (max_epsilon - min_epsilon) / decay every time step.
    If it is not, epsilon is updated as epsilon * decay. The epsilon
    value is 'clipped' to ensure that it never goes above max_epsilon
    and never below min_epsilon.
    """
    def __init__(self, max_epsilon, min_epsilon, decay):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.epsilon = max_epsilon
        self.epsilon_step = 0

    def get_epsilon(self):
        return self.epsilon

    def step(self):
        if isinstance(self.decay, int):
            delta = (self.max_epsilon - self.min_epsilon) / self.decay
            self.epsilon = np.clip(
                self.max_epsilon - delta * self.epsilon_step,
                self.min_epsilon,
                self.max_epsilon,
            )
        else:
            self.epsilon = np.clip(
                self.epsilon * self.decay, self.min_epsilon, self.max_epsilon
            )
        self.epsilon_step += 1
