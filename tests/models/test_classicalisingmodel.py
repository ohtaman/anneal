# -*- coding:utf-8 -*-

import numpy as np

from anneal.models.classicalisingmodel import ClassicalIsingModel


class TestState(object):
    def test_can_create(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = ClassicalIsingModel.State(sigma)
        assert testee.shape == sigma.shape

    def test_getitem(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = ClassicalIsingModel.State(sigma)
        assert testee[0, 1] == 1
        assert testee[1, 1] == -1

    def test_setitem(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = ClassicalIsingModel.State(sigma)
        testee[1, 1] = 1
        assert testee[1, 1] == 1

    def test_get_flatten_array(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = ClassicalIsingModel.State(sigma)
        assert (testee.get_flatten_array() == np.array([1, 1, -1, -1])).all()

    def test_to_array(self):
        sigma = np.array([[1, 1], [-1, -1]])
        testee = ClassicalIsingModel.State(sigma)
        assert (testee.to_array() == sigma).all()


class TestQUBOState(object):
    def test_can_create(self):
        testee = ClassicalIsingModel.QUBOState.random_state((5, 5))
        array = testee.to_array()
        assert testee.shape == (5, 5)
        assert ((array == 1) + (array == 0)).all()

    def test_flip_spins(self):
        testee = ClassicalIsingModel.QUBOState(np.array([[1, 1], [0, 0]]))
        testee.flip_spins(((0, 0), (1, 1)))
        assert (testee.to_array() == [[0, 1], [0, 1]]).all()


class TestIsingState(object):
    def test_can_create(self):
        testee = ClassicalIsingModel.IsingState.random_state((5, 5))
        array = testee.to_array()
        assert testee.shape == (5, 5)
        assert ((array == 1) + (array == -1)).all()

    def test_flip_spins(self):
        testee = ClassicalIsingModel.IsingState(np.array([[1, 1], [-1, -1]]))
        testee.flip_spins(((0, 0), (1, 1)))
        assert (testee.to_array() == [[-1, 1], [-1, 1]]).all()


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
        testee = ClassicalIsingModel(j, h, c, state_shape=(3,), beta=1)
        assert testee.j == j
        assert testee.h == h
        assert testee.c == c
        assert testee.beta == beta
        assert testee.neighbor_size == 3
        assert testee.state.shape == (3,)
        assert testee.state.__class__ == ClassicalIsingModel.QUBOState

    def test_energy(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        sigma = np.array([1, 1, 0])
        state = ClassicalIsingModel.QUBOState(sigma)
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
        sigma = np.array([1, 1, 0])
        state = ClassicalIsingModel.QUBOState(sigma)
        testee = ClassicalIsingModel(j, h, c, state=state, beta=beta, random=100)
        is_updated = testee.update_state()
        assert is_updated is True
        assert (testee.state.to_array() != [1, 1, 0]).any()

    def test_state(self):
        j = {
            (0, 1): 1,
            (0, 2): 2,
            (1, 2): 3
        }
        h = [1, 2, 3]
        c = 1
        beta = 1
        sigma = np.array([1, 1, 0])
        state = ClassicalIsingModel.QUBOState(sigma)
        testee = ClassicalIsingModel(j, h, c, state=state, beta=beta)
        assert testee.state == state
