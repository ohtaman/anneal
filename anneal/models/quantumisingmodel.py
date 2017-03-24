# -*- coding:utf-8 -*-

import abc
import math
import random
import sys

import numpy as np
import scipy.sparse as sp

from .physicalmodel import PhysicalModel
from .classicalisingmodel import ClassicalIsingModel


class QuantumIsingModel(PhysicalModel):
    class State(PhysicalModel.State):
        """
        State of Quantum ising model.

        Arguments:
            sigma (ndarray or list): The sigma value.
        """
        def __init__(self, sigma, classical_state_class):
            self.shape = sigma.shape[:-1]  # shape means shape of classical state.
            self.n_trotter = sigma.shape[-1]
            self._flatten = np.array(sigma).reshape(-1, self.n_trotter)
            self.ClassicalState = classical_state_class

        @abc.abstractclassmethod
        def random_state(cls, shape, n_trotter=32):
            """
            Generate random state with given shape.

            Arguments:
                shape (tuple of int): Shape of classical state.
                n_trotter (int): number of Trotter layers
            """
            pass

        def __getitem__(self, idx):
            trotter_idx = idx[-1]
            classical_idx = idx[:-1]
            flatten_idx = np.ravel_multi_index(classical_idx, self.shape)
            return self._flatten[flatten_idx, trotter_idx]

        def __setitem__(self, idx, value):
            trotter_idx = idx[-1]
            classical_idx = idx[:-1]
            flatten_idx = np.ravel_multi_index(classical_idx, self.shape)
            self._flatten[flatten_idx, trotter_idx] = value
            return self

        def __repr__(self):
                return '{}(sigma=np.array({}), classical_state_class={})'.format(
                    self.__class__.__name__,
                    self.shape,
                    self.ClassicalState
                )

        def __str__(self):
            return self.__repr__()

        def get_flatten_array(self):
            return self._flatten

        def get_trotter_layer(self, idx):
            sigma = self._flatten[:, idx].reshape(self.shape)
            return self.ClassicalState(sigma)

        def to_array(self):
            return self._flatten.reshape(self.shape + (self.n_trotter,))

        @abc.abstractmethod
        def flip_spins(self, indices):
            pass

        @property
        def size(self):
            return self._flatten[:, 0].size

    class QUBOState(State):
        def __init__(self, sigma):
            super(self.__class__, self).__init__(
                sigma,
                ClassicalIsingModel.QUBOState
            )

        @classmethod
        def random_state(cls, shape, n_trotter):
            shape_with_trotter = tuple(shape) + (n_trotter,)
            sigma = np.random.randint(0, 2, size=shape_with_trotter)
            return cls(sigma)

        def flip_spins(self, indices):
            for index in indices:
                self[index] *= -1
                self[index] += 1

    class IsingState(State):
        def __init__(self, sigma):
            super(self.__class__, self).__init__(
                sigma,
                ClassicalIsingModel.IsingState
            )

        @classmethod
        def random_state(cls, shape, n_trotter):
            shape_with_trotter = tuple(shape) + (n_trotter,)
            sigma = np.random.randint(0, 2, size=shape_with_trotter)
            return cls(2*sigma - 1)

        def flip_spins(self, indices):
            for index in indices:
                self[index] *= -1

    def __init__(self, j, h, c=0, state_type='qubo', state_shape=None, n_trotter=16, beta=10, gamma=0.1, state=None, neighbor_size=None):
        if state is None:
            assert(state_shape is not None)
            if state_type == 'qubo':
                State = self.QUBOState
            elif state_type == 'ising':
                State = self.IsingState
            else:
                raise ValueError('Unknown state type "{}"'.format(state_type))
            state = State.random_state(state_shape, n_trotter=n_trotter)
        else:
            assert(state_shape is None or state_shape == state.shape)
        if neighbor_size is None:
            neighbor_size = state.size

        self.j = j
        self.h = h
        self.c = c
        self.n_trotter = n_trotter
        self.beta = beta
        self.gamma = gamma
        self._state = state
        self.neighbor_size = neighbor_size

        if isinstance(j, dict):
            dok_flatten_j = sp.dok_matrix((self.state.size, self.state.size))
            for idx, value in j.items():
                x = np.ravel_multi_index(idx[:len(state.shape)], self.state.shape)
                y = np.ravel_multi_index(idx[len(state.shape):], self.state.shape)
                dok_flatten_j[x, y] = value
            self._flatten_j = dok_flatten_j.tocsr()
        elif isinstance(j, list):
            self._flatten_j = np.array(j).reshape(state.size, state.size)
        elif isinstance(j, np.ndarray):
            self._flatten_j = j.reshape(state.size, state.size)
        else:
            raise ValueError('QuantumIsingModel supports only dict, list and np.ndarray.')

        if isinstance(h, dict):
            self._flatten_h = np.zeros(self.state.size)
        elif isinstance(h, list):
            self._flatten_h = np.array(h).flatten()
        elif isinstance(h, np.ndarray):
            self._flatten_h = h.flatten()
        else:
            raise ValueError('QuantumIsingModel supports only dict, list and np.ndarray.')

    def __repr__(self):
        return (
            '{}('
            'j={}, '
            'h={}, '
            'c={}, '
            'state={}, '
            'beta={}, '
            'gamma={}, '
            'neighbor_size={})'
        ).format(
            self.__class__.__name__,
            str(self.j)[:10] + '...',
            str(self.h)[:10] + '...',
            self.c,
            self.state,
            self.beta,
            self.gamma,
            self.neighbor_size
        )

    def __str__(self):
        return self.__repr__()

    def energy(self, state=None):
        classical_energy = self.objective_value(state)
        quantum_energy = self.quantum_energy(state)
        return classical_energy + quantum_energy

    def objective_value(self, state=None):
        if state is None:
            state = self.state
        flatten_state = state.get_flatten_array()

        return min([
            self._classical_layer_energy(flatten_state[:, trotter_idx])
            for trotter_idx in range(self.n_trotter)
        ])

    def _classical_layer_energy(self, trotter_layer):
        return (
            - trotter_layer.dot(self._flatten_j.dot(trotter_layer))
            - self._flatten_h.dot(trotter_layer)
            - self.c
        )

    def classical_energy(self, state=None):
        if state is None:
            state = self.state
        flatten_state = state.get_flatten_array()

        return np.mean([
            self._classical_layer_energy(flatten_state[:, trotter_idx])
            for trotter_idx in range(self.n_trotter)
        ])

    def quantum_energy(self, state=None):
        if state is None:
            state = self.state
        flatten_state = state.get_flatten_array()
        e = 0
        if not self.n_trotter == 1:
            coeff = self._logcoth(self.beta*self.gamma/self.n_trotter)/(2.*self.beta)
            if self.state.__class__ == self.QUBOState:
                sigma = 2*flatten_state - 1
                e -= coeff*(sigma[:, :-1]*sigma[:, 1:]).sum()
                e -= coeff*(sigma[:, -1].dot(sigma[:, 0]))
            else:
                e -= coeff*(flatten_state[:, :-1]*flatten_state[:, 1:]).sum()
                e -= coeff*(flatten_state[:, -1].dot(flatten_state[:, 0]))
        return e

    def update_state(self):
        current_energy = self.energy()
        updated = False
        for layer in range(self.n_trotter):
            flipped = self._flip_spins(trotter_layer=layer)
            candidate_energy = self.energy()
            delta = max(0.0, candidate_energy - current_energy)
            if math.exp(-self.beta*delta) > random.random():
                updated = True
            else:
                # Cancel flipping
                self._flip_spins(flipped)
        return updated

    def _flip_spins(self, indices=None, trotter_layer=None):
        if indices is None:
            if trotter_layer is None:
                trotter_layer = np.random.randint(self.n_trotter)
            num_flip = min(
                np.random.randint(self.neighbor_size) + 1,
                self.state.size
            )
            indices = [
                np.unravel_index(
                    flatten_idx, self.state.shape
                ) + (trotter_layer,)
                for flatten_idx in np.random.choice(
                    range(self.state.size),
                    num_flip,
                    replace=False
                )
            ]
        self.state.flip_spins(indices)
        return indices

    def observe(self):
        trotter_idx = np.random.randint(self.n_trotter)
        return self.state.get_trotter_layer(trotter_idx)

    def observe_best(self):
        classical_model = ClassicalIsingModel(
            j=self.j,
            h=self.h,
            c=self.c,
            state_shape=self.state.shape
        )
        min_energy = sys.maxsize
        best_state = None
        for idx in range(self.n_trotter):
            classical_state = self.state.get_trotter_layer(idx)
            classical_energy = classical_model.energy(classical_state)
            if classical_energy < min_energy:
                min_energy = classical_energy
                best_state = classical_state
        return best_state

    @property
    def state(self):
        return self._state

    @staticmethod
    def _logcoth(x):
        return np.log(np.cosh(x)/np.sinh(x))
