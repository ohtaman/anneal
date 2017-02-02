# -*- coding:utf-8

import abc


class PhysicalModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_energy(self):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    @property
    def energy(self):
        return self.get_energy()

    @property
    def state(self):
        return self.get_state()
