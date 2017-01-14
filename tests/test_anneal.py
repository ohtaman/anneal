# -*- coding:utf-8 -*-
import numpy as np

from anneal.anneal import PhysicalModel, ClassicalIsingModel, SimulatedAnnealer


class TestPhysicalModel(object):
    def setup_method(self, method):
        class Testee(PhysicalModel):
            def get_energy(self):
                return 10

            def get_state(self):
                return [1, 0, 0]

            def update(self, **kwargs):
                return True

        self.testee = Testee()

    def test_energy(self):
        assert self.testee.energy == 10

    def test_state(self):
        assert self.testee.state == [1, 0, 0]


class TestClassicalIsingModel(object):
    def setup_method(self, method):
        self.testee = ClassicalIsingModel(
            lattice_shape=(2, 2),
            j={
                (0, 0, 1, 1): 1.0,
                (1, 1, 0, 0): 1.0,
                (0, 1, 1, 1): -1.0,
                (1, 1, 0, 1): -1.0
            },
            h=None,
            neighbor_size=2
        )

    def test_energy(self):
        state = self.testee.state
        energy = (
            1.0*state[0, 0]*state[1, 1]
            + 1.0*state[1, 1]*state[0, 0]
            - 1.0*state[0, 1]*state[1, 1]
            - 1.0*state[1, 1]*state[0, 1]
        )
        assert energy == self.testee.energy

    def test_state(self):
        state = self.testee.state
        assert state.shape == (2, 2)
        assert ((state == 1) | (state == -1)).all()

    def test_flip_spin(self):
        state = self.testee.state.copy()
        self.testee._flip_spin([0, 2])
        flipped = self.testee.state
        assert state[0, 0] == - flipped[0, 0]
        assert state[0, 1] == flipped[0, 1]
        assert state[1, 0] == - flipped[1, 0]
        assert state[1, 1] == flipped[1, 1]

    def test_update(self):
        energy = self.testee.energy
        state = self.testee.state.copy()
        for _ in range(100):
            updated = self.testee.update(beta=100)
            if updated:
                assert (state != self.testee.state).any()
                state = self.testee.state.copy()
                energy = self.testee.energy
            else:
                assert (state == self.testee.state).all()
                assert energy == self.testee.energy


class TestSimulatedAnnealer(object):
    def setup_method(self, method):
        self.model = ClassicalIsingModel(
            lattice_shape=(2, 2),
            j={
                (0, 0, 1, 1): 1.0,
                (1, 1, 0, 0): 1.0,
                (0, 1, 1, 1): -1.0,
                (1, 1, 0, 1): -1.0
            },
            h=None,
            neighbor_size=2
        )
        self.testee = SimulatedAnnealer(
            model=self.model,
            initial_temp=10.0,
            freeze_limit=10
        )

    def test_is_frozen(self):
        assert self.testee.is_frozen() == False
        self.testee.freeze_count = 10
        assert self.testee.is_frozen() == True

    def test_update(self):
        initial_energy = self.model.energy
        self.testee.anneal()
        assert initial_energy >= self.model.energy
