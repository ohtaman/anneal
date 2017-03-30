# -*- coding:utf-8 -*-

import numpy as np

from .annealer import Annealer


class QuantumAnnealer(Annealer):
    def __init__(self, model, beta_factor=1.0, gamma_factor=0.95):
        super().__init__(model)
        self.beta_factor = beta_factor
        self.gamma_factor = gamma_factor
        self.min_energy = self.model.energy()

    def __repr__(self):
        return (
            'QuantumAnnealer('
            'model={}, '
            'beta_factor={}, '
            'gamma_factor={})'
        ).format(
            self.model,
            self.beta_factor,
            self.gamma_factor
        )

    def __str__(self):
        return self.__repr__()

    def update_model(self, state_is_updated):
        if state_is_updated:
            energy = self.model.classical_energy()
            if energy < self.min_energy:
                self.min_energy = energy
        self.model.beta *= self.beta_factor
        self.model.gamma *= self.gamma_factor
