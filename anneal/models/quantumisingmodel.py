# -*- coding:utf-8 -*-

import abc
import math
import numbers
import sys

import numpy as np
import scipy.sparse as sp

from .physicalmodel import PhysicalModel


class QuantumIsingModel(PhysicalModel):
    @classmethod
    def initial_state(cls, size, n_trotter, state_type):
        if state_type == 'qubo':
            return cls.initial_qubo_state(size, n_trotter)
        elif state_type == 'ising':
            return cls.initial_ising_state(size, n_trotter)

    @classmethod
    def initial_qubo_state(cls, size, n_trotter):
        return np.random.randint(2, size=(n_trotter, size), dtype=np.int8)

    @classmethod
    def initial_ising_state(cls, size, n_trotter):
        return np.random.randint(2, size=(n_trotter, size), dtype=np.int8)*2 - 1

    def __init__(self, j, h, c=0, beta=1.0, gamma=1.0, n_trotter=16, global_flip=False, state=None, state_size=None, state_type='qubo', random_state=None):
        if state is None:
            state = self.initial_state(state_size, n_trotter, state_type)
        else:
            n_trotter, state_size = state.shape

        j = self._as_matrix(j, (state_size, state_size))
        h = self._as_matrix(h, state_size)
        j, h = self._to_triangular(j, h)
        j = sp.csr_matrix(j)
        jt = j.T.tocsr()
        j2 = j + jt

        self.j = j
        self.j2 = j2
        self.h = h
        self.c = c
        self.n_trotter = n_trotter
        self._beta = beta
        self._gamma = gamma
        self._update_coeff()
        self.global_flip = global_flip
        self._state = state
        self.state_size = state_size
        self.state_type = state_type
        self._is_qubo = 1 if state_type == 'qubo' else 0
        if isinstance(random_state, (numbers.Number, None.__class__)):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

    def __repr__(self):
        return (
            '{}('
            'j={}, '
            'h={}, '
            'c={}, '
            'beta={}, '
            'gamma={}, '
            'n_trotter={}, '
            'state={}, '
            'state_type={}, '
            'random={}'
            ')'
        ).format(
            self.__class__.__name__,
            str(self.j)[:10] + '...',
            str(self.h)[:10] + '...',
            self.c,
            self.beta,
            self.gamma,
            self.n_trotter,
            str(self.h)[:10] + '...',
            self.state_type,
            self.random_state
        )

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._update_coeff()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self._update_coeff()

    def _update_coeff(self):
        # Avoid overflow
        beta_gamma = max(np.finfo(float).eps, self._beta*self._gamma)
        self._coeff = -np.log(np.tanh(beta_gamma/self.n_trotter))/(2.*self._beta)

    @staticmethod
    def _as_matrix(list_or_dict, shape=None):
        if isinstance(list_or_dict, dict):
            matrix = np.zeros(shape)
            for (i, j), v in list_or_dict.items():
                matrix[i, j] = v
            return matrix
        else:
            return list_or_dict

    @staticmethod
    def _to_triangular(j, h):
        h = h + j.diagonal()
        j = (1 - np.tri(h.size))*(j + j.T)
        return j, h

    def _flip_spin(self, index, trotter_index):
        self._state[trotter_index, index] *= -1
        if self._is_qubo:
            self._state[trotter_index, index] += 1

    def _flip_spin_global(self, index):
        self._state[:, index] *= -1
        if self._is_qubo:
            self._state[:, index] += 1

    def energy_diff(self, index, trotter_index):
        layer = self._state[trotter_index]
        spin = layer[index]
        prev_spin = self._state[trotter_index - 1, index]
        next_spin = self._state[(trotter_index + 1)%self.n_trotter, index]

        if self._is_qubo:
            spin = spin*2 - 1
            prev_spin = prev_spin*2 - 1
            next_spin = next_spin*2 - 1

        classical_diff = spin*(
            self.j2.dot(layer)[index]
            + self.h[index]
        )/self.n_trotter

        quantum_diff = self._coeff*spin*(prev_spin + next_spin)
        return classical_diff + quantum_diff

    def energy_diff_global_flip(self, index):
        spins = self._state[:, index]
        if self._is_qubo:
            spins = spins*2 -1

        return spins.dot(self.j2.dot(self._state.T)[index] + self.h[index])

    def energy(self):
        return self.classical_energy() + self.quantum_energy()

    def _classical_layer_energy(self, trotter_index):
        trotter_layer = self._state[trotter_index]
        return (
            - self.j.dot(trotter_layer).dot(trotter_layer)
            - self.h.dot(trotter_layer)
            - self.c
        )

    def classical_energy(self):
        return np.mean([
            self._classical_layer_energy(trotter_index)
            for trotter_index in range(self.n_trotter)
        ])

    def quantum_energy(self, state=None):
        if self._is_qubo:
            spin = 2*self._state - 1
        else:
            spin = self._state

        return -self._coeff*(
            (spin[:-1]*spin[1:]).sum()
            + spin[-1].dot(spin[0])
        )

    def update_state(self):
        updated = False
        for trotter_index in range(self.n_trotter):
            indices = self.random_state.permutation(self.state_size)
            r = self.random_state.rand(self.state_size)
            for index in indices:
                delta = max(0., self.energy_diff(index, trotter_index))
                if math.exp(-self._beta*delta) > r[index]:
                    self._flip_spin(index, trotter_index)
                    updated = True

        if self.global_flip:
            indices = self.random_state.permutation(self.state_size)
            r = self.random_state.rand(self.state_size)
            for index in indices:
                delta = max(0., self.energy_diff_global_flip(index))
                if math.exp(-self._beta*delta) > r[index]:
                    self._flip_spin_global(index)
                    updated = True
        return updated

    def objective_value(self):
        return min([
            self._classical_layer_energy(i)
            for i in range(self.n_trotter)
        ])

    def observe(self):
        trotter_index = self.random_state.randint(self.n_trotter)
        return self._state[trotter_index]

    def observe_best(self):
        min_energy = sys.maxsize
        best_index = None
        for i in range(self.n_trotter):
            energy = self._classical_layer_energy(i)
            if energy < min_energy:
                min_energy = energy
                best_index = i
        return self._state[best_index]

    @property
    def state(self):
        return self._state
