# -*- coding:utf-8 -*-

import abc
import logging
import math
import random

import numpy as np
import scipy.sparse as sp


logger = logging.getLogger(__name__)


class PhysicalModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_state(self):
        pass

    @property
    def state(self):
        return self.get_state()

    @abc.abstractmethod
    def get_energy(self):
        pass

    @property
    def energy(self):
        return self.get_energy()

    @abc.abstractmethod
    def update(self, **kwargs):
        pass


class ClassicalIsingModel(PhysicalModel):
    def __init__(self, lattice_shape, j, h=None, neighbor_size=2):
        """Class of Ising model.

        Arguments:
            lattice_shape (tuple): Shape of lattice.
            j (dict): Interaction strength.
            h (np.ndarray): external field strength.
            neighbor_size (int): Size of neighbors.
        """
        super().__init__()

        self.lattice_shape = lattice_shape
        self.lattice_size = np.prod(lattice_shape)
        self.j = j
        self.h = h
        self.neighbor_size = neighbor_size

        self._j = sp.dok_matrix((self.lattice_size, self.lattice_size))
        for k, v in j.items():
            row = np.ravel_multi_index(k[len(k)//2:], lattice_shape)
            col = np.ravel_multi_index(k[:len(k)//2], lattice_shape)
            self._j[row, col] = v
        self._h = np.zeros(self.lattice_size)
        if h:
            assert h.shape == lattice_shape
            self._h = h.flatten()

        self._state = self._generate_initial_state()

    def _generate_initial_state(self):
        return np.random.randint(2, size=self.lattice_shape)*2 - 1

    def get_state(self):
        return self._state

    def get_energy(self):
        flatten = self._state.flatten()
        e = flatten.T.dot(self._j.dot(flatten))
        e += self._h.dot(flatten) if self.h else 0.
        return e

    def update(self, beta, **kwargs):
        current_energy = self.get_energy()
        num_flip = np.random.randint(self.neighbor_size) + 1
        indices = np.random.choice(
            range(self.lattice_size),
            num_flip,
            replace=False
        )

        self._flip_spin(indices)

        candidate_energy = self.get_energy()
        delta = max(0.0, candidate_energy - current_energy)

        if math.exp(-beta*delta) >= random.random():
            return True
        else:
            self._flip_spin(indices)
            return False

    def _flip_spin(self, flatten_indices):
        for idx in flatten_indices:
            value = self.state.take(idx)
            self.state.put(idx, -value)
        return self.state


class Annealer(metaclass=abc.ABCMeta):
    def __init__(self, model):
        self.model = model

    def anneal(self, **kwargs):
        return self.optimize(**kwargs)

    def optimize(self, max_iter=10000, **kwargs):
        for _ in range(max_iter):
            if self.is_frozen():
                break
            self.update(self.model.update(**self.params))

    @abc.abstractmethod
    def is_frozen(self):
        pass

    @abc.abstractmethod
    def update(self, updated):
        pass

    @abc.abstractproperty
    def params(self):
        pass


class SimulatedAnnealer(Annealer):
    def __init__(self, model, initial_temp, temp_factor=0.95, freeze_limit=10):
        super().__init__(model)

        self.temp = initial_temp
        self.temp_factor = temp_factor
        self.freeze_limit = freeze_limit
        self.freeze_count = 0
        self.min_energy = self.model.get_energy()

    def is_frozen(self):
        return self.freeze_count >= self.freeze_limit

    def update(self, updated):
        energy = self.model.energy
        if updated and energy < self.min_energy:
            self.min_energy = energy
            self.freeze_count = 0
        else:
            self.freeze_count += 1

        self.temp *= self.temp_factor

    @property
    def params(self):
        return dict(beta=1.0/self.temp)
