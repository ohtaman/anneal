# -*- coding:utf-8 -*-

import numpy as np
import scipy.sparse as sp

from anneal.models.classicalisingmodel import ClassicalIsingModel


class TestClassicalIsingModel(object):
    def test_can_create(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        testee = ClassicalIsingModel(j, h, c, state_size=3, beta=1)
        assert (testee.j == np.array([[0, 1, 2],[0, 0, 3],[0, 0, 0]])).all()
        assert (testee.h == h).all()
        assert testee.c == c
        assert testee.beta == beta
        assert testee.state.shape == (3,)
        assert testee.state_type == 'qubo'

    def test_energy(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        state = np.array([1, 1, 0])
        testee = ClassicalIsingModel(j, h, c, state=state, beta=beta)
        energy = -j[0, 1] - h[0] - h[1] - c
        assert testee.energy() == energy

    def test_update_state(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        state = np.array([1, 1, 0])
        testee = ClassicalIsingModel(j, h, c, state=state, beta=beta, random_state=2)
        is_updated = testee.update_state()
        assert is_updated is True
        assert (testee.state != [1, 1, 0]).any()
