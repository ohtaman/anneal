# -*- coding:utf-8 -*-

import abc
import math
import numbers

import numpy as np
import scipy.sparse as sp

from .physicalmodel import PhysicalModel


class ClassicalIsingModel(PhysicalModel):
    @classmethod
    def initial_state(cls, shape, state_type):
        if state_type == 'qubo':
            return cls.initial_qubo_state(shape)
        elif state_type == 'ising':
            return cls.initial_ising_state(shape)

    @classmethod
    def initial_qubo_state(cls, shape):
        return np.random.randint(2, size=shape, dtype=np.int8)

    @classmethod
    def initial_ising_state(cls, shape):
        return np.random.randint(2, size=shape, dtype=np.int8)*2 - 1

    def __init__(self, j, h, c=0, beta=1.0, state=None, state_size=None, state_type='qubo', random_state=None):
        if state is None:
            state = self.initial_state(state_size, state_type)
        self.state_size = state.size

        j = self._as_matrix(j, (self.state_size, self.state_size))
        h = self._as_matrix(h, self.state_size)
        j, h = self._to_triangular(j, h)
        j = sp.csr_matrix(j)
        jt = j.T.tocsr()

        self.j = j
        self.jt = jt
        self.h = h
        self.c = c
        self.beta = beta
        self._state = state
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
            str(self.h)[:10] + '...',
            self.state_type,
            self.random_state
        )

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
        h = h + j.diagonal()**2
        j = (1 - np.tri(h.size))*(j + j.T)
        return j, h

    def _flip_spin(self, index):
        self._state[index] *= -1
        if self._is_qubo:
            self._state[index] += 1

    def energy_diff(self, index):
        return (1 if self._state[index] > 0 else -1)*(
            self.j[index].dot(self._state)[0]
            + self.jt[index].dot(self._state)[0]
            + self.h[index]
        )

    def energy(self):
        e = -self.c
        e -= self.j.dot(self._state).dot(self._state)
        e -= self.h.dot(self._state)
        return e

    def update_state(self):
        updated = False
        indices = self.random_state.permutation(self.state_size)

        for index in indices:
            delta = max(0., self.energy_diff(index))
            if math.exp(-self.beta*delta) > self.random_state.rand():
                self._flip_spin(index)
                updated = True
        return updated

    @property
    def state(self):
        return self._state
