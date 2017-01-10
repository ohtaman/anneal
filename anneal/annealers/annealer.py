# -*- coding:utf-8 -*-

import abc


class Annealer(metaclass=abc.ABCMeta):
    def initialize(self, initial_state):
        return initial_state

    def on_accept(self, candidate_state, candidate_energy):
        pass

    def on_reject(self, candidate_state, candidate_energy):
        pass

    @abc.abstractmethod
    def energy(self, state):
        pass

    @abc.abstractmethod
    def get_neighbor(self, state):
        pass

    @abc.abstractmethod
    def is_acceptable(self, candidate_state, candidate_energy):
        pass

    @abc.abstractmethod
    def is_frozen(self):
        pass

    def anneal(self, **kwargs):
        return self.optimize(**kwargs)

    def optimize(self, initial_state=None):
        self.current_state = self.initialize(initial_state)
        self.current_energy = self.energy(self.current_state)
        self.iter_step = 0

        while not self.is_frozen():
            candidate_state = self.get_neighbor(self.current_state)
            candidate_energy = self.energy(candidate_state)
            if self.is_acceptable(candidate_state, candidate_energy):
                self.current_state = candidate_state
                self.current_energy = candidate_energy

            self.iter_step += 1

        return self.state
