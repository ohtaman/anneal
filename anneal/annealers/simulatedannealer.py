# -*- coding:utf-8 -*-

import math
import random

from .annealer import Annealer


class BasicSimulatedAnnealer(Annealer):
    def __init__(self, energy_fn, neighbor_fn, max_iter=1000, initial_temperature=100000):
        self.energy_fn = energy_fn
        self.neighbor_fn = neighbor_fn
        self.max_iter = max_iter
        self.initial_temperature = initial_temperature

    def temperature(self):
        return self.initial_temperature**(1/self.iter_step)

    def get_energy(self, state):
        temperature = self.temperature_fn(self)
        return self.energy(state, temperature)

    def get_neighbor(self, state):
        return self.neighbor_fn(state)

    def is_acceptable(self, candidate_state, candidate_energy):
        delta = max(0.0, candidate_energy - self.current_energy)
        return math.exp(-delta) >= random.random()

    def is_frozen(self):
        return self.iter_step >= self.max_iter
