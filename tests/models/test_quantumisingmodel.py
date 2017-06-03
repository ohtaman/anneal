# -*- coding:utf-8 -*-

import numpy as np
import scipy.sparse as sp
import pytest

from anneal.models.quantumisingmodel import QuantumIsingModel


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
        testee = QuantumIsingModel(j, h, c, state_size=3, beta=beta, gamma=gamma)
        assert (testee.j == np.array([[0, 1, 2],[0, 0, 3],[0, 0, 0]])).all()
        assert (testee.h == h).all()
        assert testee.c == c
        assert testee.beta == beta
        assert testee.gamma == gamma
        assert testee.state.shape == (testee.n_trotter, 3)
        assert testee.state_type == 'qubo'

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
        state = np.array([[1, 1, 0], [1, 0, 1]])
        n_trotter = state.shape[0]
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
        state = np.array([[1, 1, 0], [1, 0, 1]])
        testee = QuantumIsingModel(j, h, c, state=state.copy(), beta=beta, random_state=2)
        is_updated = testee.update_state()
        assert is_updated is True
        assert (testee.state != state).any()
