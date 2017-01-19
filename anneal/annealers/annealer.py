# -*- coding:utf-8 -*-

import abc
import logging


logger = logging.getLogger(__name__)


class Annealer(metaclass=abc.ABCMeta):
    def __init__(self, model):
        self.iter_count = 0
        self.model = model

    def anneal(self, **kwargs):
        """ Alias of optimize method. """
        return self.optimize(**kwargs)

    def optimize(self, max_iter=None, iter_callback=None):
        """ Minimize energy of self.model.

        Args:
            max_iter (int): Maximum number of iterations
            iter_callback (callable): Callback function which called on each iteration.

        Returns:
            bool: True if the state is frozen, False if iteration count exceeds.
        """
        iter_count = 0
        while not self.is_frozen():
            if max_iter is not None and iter_count > max_iter:
                break

            state_is_updated = self.model.update()
            self.update(state_is_updated)
            self.iter_count += 1
            iter_count += 1
            if iter_callback:
                iter_callback(self, state_is_updated)
        else:
            pass
        return self.is_frozen()

    @abc.abstractmethod
    def is_frozen(self):
        """ Frozen function.

        Returns:
            bool: True if the state is frozen, False if not.
        """
        pass

    @abc.abstractmethod
    def update(self, state_is_updated):
        """ Update model parameters.

        Args:
            state_is_updated (bool): True if the model state is updated, False if not.

        Returns:
            bool: True if model parameters are changed, False if not.
        """
        pass
