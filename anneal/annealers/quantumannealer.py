# -*- coding:utf-8 -*-

import numpy as np

from .annealer import Annealer


class QuantumAnnealer(Annealer):
    def __init__(self, model, beta_factor=1.0, gamma_factor=0.99, time_scale=100, wait_count=1000, freeze_limit=1000):
        super().__init__(model)

        self.beta_factor = beta_factor
        self.gamma_factor = gamma_factor
        self.time_scale = time_scale
        self.wait_count = wait_count
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
            'freeze_limit={}, '
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
            energy = self.model.energy()
            if energy < self.min_energy:
                self.min_energy = energy
                self._freeze_count = 0
        else:
            self._freeze_count += 1

        if self.iter_count%self.time_scale == 0:
            self.model.beta /= self.beta_factor
        t = max(0, self.iter_count - self.wait_count)/self.time_scale
        self.model.gamma = self._gamma_zero*np.exp(-self.gamma_factor**(-t))
