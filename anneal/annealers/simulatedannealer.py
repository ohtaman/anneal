# -*- coding:utf-8 -*-

from .annealer import Annealer


class SimulatedAnnealer(Annealer):
    def __init__(self, model, beta_factor=0.99, time_scale=100, freeze_limit=1000):
        super().__init__(model)

        self.beta_factor = beta_factor
        self.time_scale = time_scale
        self.freeze_limit = freeze_limit
        self._freeze_count = 0
        self.min_energy = self.model.energy()

    def __repr__(self):
        return (
            'SimulatedAnnealer('
            'model={}, '
            'beta_factor={}, '
            'freeze_limit={}, '
        ).format(
            self.model,
            self.beta_factor,
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
        else:
            self._freeze_count += 1

        if self.iter_count%self.time_scale == 0:
            self.model.beta /= self.beta_factor
