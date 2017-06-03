# -*- coding:utf-8 -*-

from .annealer import Annealer


class SimulatedAnnealer(Annealer):
    def __init__(self, model):
        super().__init__(model)
        self.initial_beta = model.beta

    def __repr__(self):
        return (
            'SimulatedAnnealer('
            'model={})'
        ).format(
            self.model
        )

    def __str__(self):
        return self.__repr__()

    def update_model(self, state_is_updated):
        self.model.beta = self.initial_beta*self._max_iter/(self._max_iter - self.iter_count)
