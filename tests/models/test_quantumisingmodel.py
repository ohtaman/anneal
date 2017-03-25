# -*- coding:utf-8 -*-

import numpy as np
import pytest

from anneal.models.quantumisingmodel import QuantumIsingModel


class TestState(object):
    def test_can_create(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = QuantumIsingModel.State(sigma, None)
        assert testee.shape == sigma.shape[:-1]

    def test_getitem(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = QuantumIsingModel.State(sigma, None)
        assert testee[0, 1] == 1
        assert testee[1, 1] == -1

    def test_setitem(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = QuantumIsingModel.State(sigma, None)
        testee[1, 1] = 1
        assert testee[1, 1] == 1

    def test_get_flatten_array(self):
        sigma = np.array([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]])
        testee = QuantumIsingModel.State(sigma, None)
        assert (testee.get_flatten_array() == sigma.reshape((-1, 2))).all()

    def test_to_array(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = QuantumIsingModel.State(sigma, None)
        assert (testee.to_array() == sigma).all()


class TestQUBOState(object):
    def test_can_create(self):
        testee = QuantumIsingModel.QUBOState.random_state((5, 5), 2)
        array = testee.to_array()
        assert testee.shape == (5, 5)
        assert ((array == 1) + (array == 0)).all()

    def test_flip_spins(self):
        sigma = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        flip_idx = ((0, 0, 0), (1, 1, 1))
        flipped = np.array([[[0, 1], [1, 1]], [[0, 0], [0, 1]]])
        testee = QuantumIsingModel.QUBOState(sigma)
        testee.flip_spins(flip_idx)
        assert (testee.to_array() == flipped).all()


class TestIsingState(object):
    def test_can_create(self):
        testee = QuantumIsingModel.IsingState.random_state((5, 5), 2)
        array = testee.to_array()
        assert testee.shape == (5, 5)
        assert ((array == 1) + (array == -1)).all()

    def test_flip_spins(self):
        sigma = np.array([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]])
        flip_idx = ((0, 0, 0), (1, 1, 1))
        flipped = np.array([[[-1, 1], [1, 1]], [[-1, -1], [-1, 1]]])
        testee = QuantumIsingModel.IsingState(sigma)
        testee.flip_spins(flip_idx)
        assert (testee.to_array() == flipped).all()


class TestQuantumIsingModel(object):
    def test_can_create(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        gamma = 2
        testee = QuantumIsingModel(j, h, c, state_shape=(3,), beta=beta, gamma=gamma)
        assert testee.j == j
        assert testee.h == h
        assert testee.c == c
        assert testee.beta == beta
        assert testee.gamma == gamma
        assert testee.neighbor_size == 3
        assert testee.state.shape == (3,)
        assert testee.state.__class__ == QuantumIsingModel.QUBOState

    def test_energy(self):
        j = {
            (0, 1): 1.,
            (0, 2): 2.,
            (1, 2): 3.
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        gamma = 2
        sigma = np.array([[1, 1], [1, 0], [0, 1]])
        n_trotter = sigma.shape[-1]
        state = QuantumIsingModel.QUBOState(sigma)
        testee = QuantumIsingModel(j, h, c, state=state, beta=beta, gamma=gamma)
        # When gamma == 0, energy must just be average energy of trotter layers.
        classical_energy = (
            (-j[0, 1] - h[0] - h[1] - c)
            + (-j[0, 2] - h[0] -h[2] - c)
        )/2
        quantum_coeff = np.log(np.tanh(beta*gamma/n_trotter))/(2*beta)
        quantum_energy = quantum_coeff*(1 - 1 - 1)*2
        energy = classical_energy + quantum_energy
        assert testee.classical_energy() == pytest.approx(classical_energy)
        assert testee.quantum_energy() == pytest.approx(quantum_energy)
        assert testee.energy() == pytest.approx(energy)

    def test_update_state(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        sigma = np.array([[1, 1], [1, 0], [0, 1]])
        state = QuantumIsingModel.QUBOState(sigma)
        testee = QuantumIsingModel(j, h, c, state=state, beta=beta, random=100)
        is_updated = testee.update_state()
        assert is_updated is True
        assert (testee.state.to_array() != sigma).any()

    def test_state(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        sigma = np.array([[1, 1], [1, 0], [0, 1]])
        state = QuantumIsingModel.QUBOState(sigma)
        testee = QuantumIsingModel(j, h, c, state=state, beta=beta)
        assert testee.state == state
