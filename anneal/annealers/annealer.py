# -*- coding:utf-8 -*-

import abc


class Annealer(metaclass=abc.ABCMeta):
    def __init__(self, model):
        self.iter_count = 0
        self.model = model

    def anneal(self, **kwargs):
        """ Alias of self.optimize. """
        return self.optimize(**kwargs)

    def optimize(self, max_iter=1000, iter_callback=None):
        """ Minimize the energy of self.model by annealing.

        Args:
            max_iter (int): Maximum number of iterations. Defaults to 1000.
            iter_callback (callable): Callback function which called on each iteration.

        Returns:
            bool: True if the state is frozen, False if iteration count exceeds.
        """
        while (not self.is_frozen()) and (self.iter_count < max_iter):
            state_is_updated = self.model.update_state()
            model_is_updated = self.update_model(state_is_updated)
            self.iter_count += 1
            if iter_callback is not None:
                iter_callback(self, state_is_updated, model_is_updated)

        return self.is_frozen()

    def is_frozen(self):
        """ Frozen function.

        Returns:
            bool: True if the state is frozen, False if not.
        """
        return False

    @abc.abstractmethod
    def update_model(self, state_is_updated):
        """ Update model parameters.

        Args:
            state_is_updated (bool): True if the model state is updated, False if not.

        Returns:
            bool: True if model parameters are changed, False if not.
        """
        pass
