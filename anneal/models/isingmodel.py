# -*- coding:utf-8 -*-

import math
import random

import numpy as np
import scipy.sparse as sp

from .physicalmodel import PhysicalModel


class IsingModel(PhysicalModel):
    def __init__(self, lattice_shape, j, h=None, beta=0.001, neighbor_size=3):
        """Class of Ising model.

        Arguments:
            lattice_shape (tuple): Shape of lattice.
            j (dict): Interaction strength.
            h (np.ndarray): external field strength.
            beta (float): inverse temperature.
            neighbor_size (int): Size of neighbors.
        """
        super().__init__()

        self.lattice_shape = lattice_shape
        self.lattice_size = np.prod(lattice_shape)
        self.j = j
        self.h = h
        self.beta = beta
        self.neighbor_size = neighbor_size

        # J for flatten state
        self._flat_j = sp.dok_matrix((self.lattice_size, self.lattice_size))
        for k, v in j.items():
            x = np.ravel_multi_index(k[len(k)//2:], lattice_shape)
            y = np.ravel_multi_index(k[:len(k)//2], lattice_shape)
            self._flat_j[x, y] = self._flat_j[y, x] = v  # j must be symmetric.

        # H for flatten state
        self._flat_h = np.zeros(self.lattice_size)
        if h is not None:
            assert h.shape == lattice_shape
            self._flat_h = h.flatten()

        self._state = self.generate_initial_state()

    def generate_initial_state(self):
        return np.random.randint(2, size=self.lattice_shape)*2 - 1

    def get_state(self):
        return self._state

    def get_energy(self):
        flatten = self._state.ravel()
        e = flatten.T.dot(self._flat_j.dot(flatten))
        e += self._flat_h.dot(flatten) if self.h is not None else 0.
        return e

    def update(self):
        assert hasattr(self, 'beta')

        current_energy = self.get_energy()
        num_flip = np.random.randint(self.neighbor_size) + 1
        indices = np.random.choice(
            range(self.lattice_size),
            num_flip,
            replace=False
        )

        self._flip_spins(indices)

        candidate_energy = self.get_energy()
        delta = max(0.0, candidate_energy - current_energy)

        if math.exp(-self.beta*delta) >= random.random():
            return True
        else:
            # revert flipped spins
            self._flip_spins(indices)
            return False

    def _flip_spins(self, flatten_indices):
        for idx in flatten_indices:
            value = self.state.take(idx)
            self.state.put(idx, -value)
        return self.state
