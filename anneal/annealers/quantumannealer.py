# -*- coding:utf-8 -*-

import numpy as np

from .annealer import Annealer


class QuantumAnnealer(Annealer):
    def __init__(self, model):
        super().__init__(model)
        self.initial_gamma = model.gamma

    def __repr__(self):
        return (
            'QuantumAnnealer('
            'model={})'
        ).format(
            self.model
        )

    def __str__(self):
        return self.__repr__()

    def update_model(self, state_is_updated):
        self.model.gamma = self.initial_gamma*(self._max_iter - self.iter_count)/self._max_iter
