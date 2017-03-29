# -*- coding:utf-8 -*-

import numpy as np

from .annealer import Annealer


class QuantumAnnealer(Annealer):
    def __init__(self, model, beta_factor=1.0, gamma_factor=1.05, freeze_limit=10):
        super().__init__(model)

        self.beta_factor = beta_factor
        self.gamma_factor = gamma_factor
        self.freeze_limit = freeze_limit
        self._freeze_count = 0
        self.min_energy = self.model.energy()
        self._gamma_zero = self.model.gamma

    def __repr__(self):
        return (
            'QuantumAnnealer('
            'model={}, '
            'beta_factor={}, '
            'gamma_factor={}, '
            'freeze_limit={})'
        ).format(
            self.model,
            self.beta_factor,
            self.gamma_factor,
            self.freeze_limit
        )

    def __str__(self):
        return self.__repr__()

    def is_frozen(self):
        return self._freeze_count >= self.freeze_limit

    def update_model(self, state_is_updated):
        if state_is_updated:
            energy = self.model.classical_energy()
            if energy < self.min_energy:
                self.min_energy = energy
                self._freeze_count = 0
            else:
                self._freeze_count += 1
        else:
            self._freeze_count += 1

        self.model.beta *= self.beta_factor
        self.model.gamma = self._gamma_zero*np.exp(-self.gamma_factor**self.iter_count)
