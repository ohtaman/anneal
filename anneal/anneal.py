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

    @abc.abstractmethod
    def get_energy(self, **kwargs):
        pass


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
            self._j[row, col] = self._j[col, row] = v  # j must be symmetric.
        self._h = np.zeros(self.lattice_size)
        if h:
            assert h.shape == lattice_shape
            self._h = h.flatten()

        self._state = self._generate_initial_state(lattice_shape)

    def _generate_initial_state(self, lattice_shape):
        return np.random.randint(2, size=lattice_shape)*2 - 1

    def get_state(self):
        return self._state

    def get_energy(self, **kwargs):
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


class QuantumIsingModel(PhysicalModel):
    def __init__(self, lattice_shape, j, h=None, trotter_size=16, neighbor_size=2):
        """Class of Ising model.

        Arguments:
            lattice_shape (tuple): Shape of lattice.
            j (dict): Interaction strength.
            h (np.ndarray): external field strength.
            trotter_size (int): number of Trotter slice.
            neighbor_size (int): Size of neighbors.
        """
        super().__init__()

        self.lattice_shape = lattice_shape
        self.lattice_size = np.prod(lattice_shape)
        self.j = j
        self.h = h
        self.trotter_size = trotter_size
        self.neighbor_size = neighbor_size

        self._j = sp.dok_matrix((self.lattice_size, self.lattice_size))
        for k, v in j.items():
            row = np.ravel_multi_index(k[len(k)//2:], lattice_shape)
            col = np.ravel_multi_index(k[:len(k)//2], lattice_shape)
            self._j[row, col] = self._j[col, row] = v  # j must be symmetric.
        self._h = np.zeros(self.lattice_size)
        if h:
            assert h.shape == lattice_shape
            self._h = h.flatten()

        self._state = self._generate_initial_state(
            self.lattice_shape,
            self.trotter_size
        )

    def _generate_initial_state(self, lattice_shape, trotter_size):
        shape = self.lattice_shape + (trotter_size,)
        return np.random.randint(2, shape)*2 - 1

    def get_state(self):
        return self._state

    def get_energy(self, beta, gamma):
        # Classical term
        # Flatten except trotter dimension.
        flatten = self._state.flatten()[:, self.trotter_size]
        e = flatten.T.dot(self._j.dot(flatten)).mean()
        e += self._h.dot(flatten).mean() if self.h else 0.

        # Quantum term
        coeff = (1/(2*beta))*self._logcoth(beta*gamma/self.trotter_size)

        return e

    def update(self, beta, gamma):
        current_energy = self.get_energy(beta, gamma)
        num_flip = np.random.randint(self.neighbor_size) + 1
        indices = np.random.choice(
            range(self.lattice_size),
            num_flip,
            replace=False
        )

        self._flip_spin(indices)

        candidate_energy = self.get_energy(beta, gamma)
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

    @staticmethod
    def _logcoth(x):
        return math.log(math.cosh(x)/math.sinh(x))


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
    def __init__(self, model, initial_beta, beta_factor=0.95, freeze_limit=10):
        super().__init__(model)

        self.beta = initial_beta
        self.beta_factor = beta_factor
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

        self.beta /= self.beta_factor

    @property
    def params(self):
        return dict(beta=self.beta)
