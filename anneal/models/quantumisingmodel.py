# -*- coding:utf-8 -*-

import math
import random
import numpy as np

from .isingmodel import IsingModel


class QuantumIsingModel(IsingModel):
    def __init__(self, lattice_shape, j, h=None, beta=1., gamma=10.0, n_trotter=32, neighbor_size=2):
        self.gamma = gamma
        self.n_trotter = n_trotter
        super().__init__(lattice_shape, j, h, beta, neighbor_size)

    def generate_initial_state(self):
        _super = super()
        return np.array([_super.generate_initial_state() for _ in range(self.n_trotter)])

    @property
    def classical_state(self):
        return self.get_best_layer()

    def get_best_layer(self):
        min_energy = np.inf
        best_layer = None
        flatten = self._state.reshape((-1, self.n_trotter))
        for i in range(self.n_trotter):
            layer = flatten[:, i]
            energy = self._get_classical_energy(layer)
            if energy <= min_energy:
                min_energy = energy
                best_layer = layer
        return best_layer.reshape(self.lattice_shape)

    @property
    def classical_energy(self):
        flatten = self.classical_state.ravel()
        return self._get_classical_energy(flatten)

    def _get_classical_energy(self, flatten_state):
        e = flatten_state.T.dot(self._flat_j.dot(flatten_state)).mean()
        e += self._flat_h.dot(flatten_state).mean() if self.h is not None else 0.
        return e

    def _get_quantum_energy(self, flatten_state, beta, gamma):
        n_trotter = flatten_state.shape[-1]
        coeff = self._logcoth(beta*gamma/n_trotter)/2.
        return coeff*(
            (flatten_state[:, :-1]*flatten_state[:, 1:]).sum()
            + (flatten_state[:, -1]*flatten_state[:, 0]).sum()
        )

    def get_energy(self):
        flatten = self._state.reshape((-1, self.n_trotter))
        e = self._get_classical_energy(flatten)
        e += self._get_quantum_energy(flatten, self.beta, self.gamma)
        return e

    def update(self):
        assert hasattr(self, 'beta')
        assert hasattr(self, 'gamma')

        current_energy = self.get_energy()
        num_flip = (np.random.randint(self.neighbor_size) + 1)*self.n_trotter
        indices = np.random.choice(
            range(self.lattice_size*self.n_trotter),
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

    @staticmethod
    def _logcoth(x):
        return np.log(np.cosh(x)/np.sinh(x))