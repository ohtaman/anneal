# -*- coding:utf-8 -*-

from .annealer import Annealer


class SimulatedAnnealer(Annealer):
    def __init__(self, model, beta_factor=1.05):
        super().__init__(model)

        self.beta_factor = beta_factor
        self.min_energy = self.model.energy()

    def __repr__(self):
        return (
            'SimulatedAnnealer('
            'model={}, '
            'beta_factor={})'
        ).format(
            self.model,
            self.beta_factor
        )

    def __str__(self):
        return self.__repr__()

    def update_model(self, state_is_updated):
        if state_is_updated:
            energy = self.model.energy()
            if energy < self.min_energy:
                self.min_energy = energy

        self.model.beta *= self.beta_factor
