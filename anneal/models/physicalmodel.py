# -*- coding:utf-8

import abc


class PhysicalModel(object, metaclass=abc.ABCMeta):
    class State(object):
        pass

    def objective_value(self):
        return self.energy()

    @abc.abstractmethod
    def energy(self, state=None):
        pass

    @abc.abstractmethod
    def update_state(self):
        pass

    @property
    @abc.abstractmethod
    def state(self):
        pass
