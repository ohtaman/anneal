# -*- coding:utf-8 -*-

import abc
import math
import random


class Annealer(metaclass=abc.ABCMeta):
    def initialize(self, initial_state):
        self.step_count = 0
        self.current_state = initial_state
        self.current_energy = self.energy(initial_state)

    def count_up(self, next_state, next_energy, accept):
        self.step_count += 1

    def is_acceptable(self, candidate_energy):
        delta = max(0.0, candidate_energy - self.current_energy)
        return math.exp(-delta/self.get_temperature()) >= random.random()

    def set_state(self, state, energy=None):
        if energy is None:
            energy = self.get_energy(state)
        self.current_state = state
        self.current_energy = energy
 
    @abc.abstractmethod
    def get_energy(self, state):
        pass

    @abc.abstractmethod
    def get_neighbor(self, state):
        pass

    @abc.abstractmethod
    def get_temperature(self):
        pass

    @abc.abstractmethod
    def is_frozen(self):
        pass

    def anneal(self, **kwargs):
        return self.optimize(**kwargs)

    def optimize(self, initial_state=None):
        self.initialize(initial_state)

        while not self.is_frozen():
            candidate_state = self.get_neighbor(self.current_state)
            candidate_energy = self.get_energy(candidate_state)
            accept = self.is_acceptable(candidate_energy)
            self.count_up(candidate_state, candidate_energy, accept)
            if accept:
                self.set_state(candidate_state, candidate_energy)

        return self.state
