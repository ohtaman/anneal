# -*- coding:utf-8 -*-

import abc
import math
import random

import numpy as np
import scipy.sparse as sp

from .physicalmodel import PhysicalModel


class ClassicalIsingModel(PhysicalModel):
    class State(PhysicalModel.State):
        """
        State of classical ising model.

        Arguments:
            sigma (ndarray or list): The sigma value.
        """

        def __init__(self, sigma):
            self._flatten = np.array(sigma).flatten()
            self.shape = sigma.shape

        @abc.abstractclassmethod
        def random_state(cls, shape, random=None):
            """
            Generate random state with given shape.

            Arguments:
                shape (tuple of int): Shape of state.
            """
            pass

        def __getitem__(self, idx):
            flatten_idx = np.ravel_multi_index(idx, self.shape)
            return self._flatten[flatten_idx]

        def __setitem__(self, idx, value):
            flatten_idx = np.ravel_multi_index(idx, self.shape)
            self._flatten[flatten_idx] = value
            return self

        def __repr__(self):
            return '{}(sigma=np.array({}))'.format(
                self.__class__.__name__,
                self.shape
            )

        def __str__(self):
            return self.__repr__()

        def get_flatten_array(self):
            return self._flatten

        def to_array(self):
            return self._flatten.reshape(self.shape)

        @abc.abstractmethod
        def flip_spins(self, indices):
            pass

        @property
        def size(self):
            return self._flatten.size

    class QUBOState(State):
        @classmethod
        def random_state(cls, shape, random=None):
            if random is None:
                random = np.random
            return cls(random.randint(0, 2, size=shape))

        def flip_spins(self, indices):
            for index in indices:
                flatten_idx = np.ravel_multi_index(index, self.shape)
                self._flatten[flatten_idx] *= -1
                self._flatten[flatten_idx] += 1

    class IsingState(State):
        @classmethod
        def random_state(cls, shape, random=None):
            if random is None:
                random = np.random
            return cls(2*random.randint(0, 2, size=shape) - 1)

        def flip_spins(self, indices):
            for index in indices:
                flatten_idx = np.ravel_multi_index(index, self.shape)
                self._flatten[flatten_idx] *= -1

    def __init__(self, j, h, c=0, state_type='qubo', state_shape=None, beta=1, state=None, neighbor_size=None, random=None):
        if state is None:
            assert(state_shape is not None)
            if state_type == 'qubo':
                State = self.QUBOState
            elif state_type == 'ising':
                State = self.IsingState
            else:
                raise ValueError('Unknown state type "{}"'.format(state_type))
            state = State.random_state(state_shape)
        else:
            assert(state_shape is None or state_shape == state.shape)
        if neighbor_size is None:
            neighbor_size = state.size

        self.j = j
        self.h = h
        self.c = c
        self.beta = beta
        self._state = state
        self.neighbor_size = neighbor_size
        if isinstance(random, np.random.RandomState):
            self.random_state = random
        else:
            self.random_state = np.random.RandomState(random)

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
            raise ValueError('Only dict or is supported.')

        if isinstance(h, dict):
            self._flatten_h = np.zeros(self.state.size)
        elif isinstance(h, list):
            self._flatten_h = np.array(h).flatten()
        elif isinstance(h, np.ndarray):
            self._flatten_h = h.flatten()

    def __repr__(self):
        return (
            '{}('
            'j={}, '
            'h={}, '
            'c={}, '
            'beta={}, '
            'state={}, '
            ')'
        ).format(
            self.__class__.__name__,
            str(self.j)[:10] + '...',
            str(self.h)[:10] + '...',
            self.c,
            self.beta,
            self.state
        )

    def energy(self, state=None):
        if state is None:
            state = self.state
        flatten_state = state.get_flatten_array()
        e = -self.c
        e -= flatten_state.dot(self._flatten_j.dot(flatten_state))
        e -= self._flatten_h.dot(flatten_state)
        return e

    def update_state(self):
        current_energy = self.energy()
        flipped = self._flip_spins()
        candidate_energy = self.energy()
        delta = max(0.0, candidate_energy - current_energy)
        if math.exp(-self.beta*delta) > self.random_state.rand():
            return True
        else:
            # Cancel flipping
            self._flip_spins(flipped)
            return False

    def _flip_spins(self, indices=None):
        if indices is None:
            num_flip = min(self.random_state.randint(self.neighbor_size) + 1, self.state.size)
            indices = [
                np.unravel_index(flatten_index, self.state.shape)
                for flatten_index in self.random_state.choice(
                    range(self.state.size),
                    num_flip,
                    replace=False
                )
            ]
        self.state.flip_spins(indices)
        return indices

    @property
    def state(self):
        return self._state
