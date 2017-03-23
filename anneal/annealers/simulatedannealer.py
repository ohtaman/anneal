# -*- coding:utf-8 -*-

from .annealer import Annealer


class SimulatedAnnealer(Annealer):
    def __init__(self, model, beta_factor=0.99, freeze_limit=100, update_limit=10):
        super().__init__(model)

        self.beta_factor = beta_factor
        self.freeze_limit = freeze_limit
        self.update_limit = update_limit
        self._freeze_count = 0
        self._update_count = 0
        self.min_energy = self.model.energy()

    def __repr__(self):
        return (
            'SimulatedAnnealer('
            'model={}, '
            'beta_actor={}, '
            'freeze_limit={}, '
            'update_limit={})'
        ).format(
            self.model,
            self.beta_factor,
            self.freeze_limit,
            self.update_limit
        )

    def __str__(self):
        return self.__repr__()

    def is_frozen(self):
        return self._freeze_count >= self.freeze_limit

    def update_model(self, state_is_updated):
        energy = self.model.energy()
        if state_is_updated:
            self._update_count += 1
            if energy < self.min_energy:
                self.min_energy = energy
                self._freeze_count = 0
        else:
            self._freeze_count += 1

        if self._update_count >= self.update_limit:
            self.model.beta /= self.beta_factor
            self._update_count += 1
