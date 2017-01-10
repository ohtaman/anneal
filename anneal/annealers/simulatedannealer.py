# -*- coding:utf-8 -*-

import math
import random

from .annealer import Annealer


class BasicSimulatedAnnealer(Annealer):
    def __init__(self, max_iter=10000, energy_fn, temperature_fn, get_neighbor_fn):
        self.max_iter = max_iter
        self.energy_fn = energy_fn
        self.temperature_fn = temperature_fn
        self.get_neighbor_fn = get_neighbor_fn

    def energy(self, state):
        temperature = self.temperature_fn(self)
        return self.energy(state, temperature)

    def get_neighbor(self, state):
        return self.get_neighbor_fn(state)

    def is_acceptable(self, candidate_state, candidate_energy):
        delta = max(0.0, candidate_energy - self.current_energy)
        return math.exp(-delta) >= random.random()

    def is_frozen(self):
        return self.iter_step >= self.max_iter
