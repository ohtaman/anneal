# -*- coding:utf-8 -*-

import collections
import sys
import time

import numpy as np
import scipy.sparse as sp

from anneal.models import ClassicalIsingModel, QuantumIsingModel
from anneal.annealers import SimulatedAnnealer, QuantumAnnealer


POSITIONS = [
    (24050.0000, 123783),
    (24216.6667, 123933),
    (24233.3333, 123950),
    (24233.3333, 124016),
    # (24250.0000, 123866),
    # (24300.0000, 123683),
    # (24316.6667, 123900),
    # (24316.6667, 124083),
    # (24333.3333, 123733),
    # (24333.3333, 123983),
    # (24333.3333, 124150),
    # (24333.3333, 124200),
    # (24350.0000, 123733),
    # (24350.0000, 123750),
    # (24350.0000, 124216),
    # (24350.0000, 124233),
    # (24383.3333, 123750),
    # (24383.3333, 124150),
    # (24400.0000, 123833),
    # (24416.6667, 123766),
    # (24416.6667, 124250),
    # (24433.3333, 122983),
    # (24450.0000, 122933),
    # (24450.0000, 124133),
    # (24450.0000, 124183),
    # (24466.6667, 123000),
    # (24500.0000, 124283),
    # (24583.3333, 124316),
    # (24666.6667, 124700),
    # (24716.6667, 125333),
    # (24733.3333, 125283),
    # (24733.3333, 125316),
    # (24733.3333, 125350),
    # (24733.3333, 125400),
    # (24733.3333, 125416),
    # (24750.0000, 125266),
    # (24750.0000, 125283),
    # (24766.6667, 125366),
    # (24783.3333, 125266),
    # (24783.3333, 125300),
    # (24783.3333, 141316),
    # (24783.3333, 141333),
    # (24800.0000, 125166),
    # (24800.0000, 125283),
    # (24800.0000, 141300),
    # (24800.0000, 141316),
    # (24800.0000, 141333),
    # (24816.6667, 125166),
    # (24816.6667, 125300),
    # (24833.3333, 125166),
]


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(((a - b)**2).sum())


def build_weights(positions, coeff=500):
    n_cities = len(positions)

    j = collections.defaultdict(int)
    max_dist = 0
    for t in range(n_cities):
        for a in range(n_cities):
            for b in range(n_cities):
                d = dist(positions[a], positions[b])
                max_dist = d if max_dist < d else max_dist
                j[(a, t, b, (t + 1)%n_cities)] = -d

    A = max_dist * coeff
    for t in range(n_cities):
        for a in range(n_cities):
            for b in range(n_cities):
                if a != b:
                    j[(a, t, b, t)] -= 2*A

    for a in range(n_cities):
        for t1 in range(n_cities):
            for t2 in range(n_cities):
                if t1 != t2:
                    j[(a, t1, a, t2)] -= 2*A

    h = np.zeros((n_cities, n_cities))
    for t in range(n_cities):
        for a in range(n_cities):
            h[a, t] += 2*A

    c = -2*A*n_cities
    return j, h, c


def callback(annealer, state_is_updated, model_is_updated):
    if annealer.iter_count % 100 == 0:
        print("{}: {}'th iter.".format(
            annealer.__class__.__name__,
            annealer.iter_count
        ))


def check_constraints(state):
    return (
        (state.to_array().sum(axis=1) == 1).all()
        and (state.to_array().sum(axis=0) == 1).all()
    )

def solve_tsp():
    j, h, c = build_weights(POSITIONS, 500)

    start = time.time()
    min_energy = float('inf')
    best_annealer = None
    iter = 0
    for i in range(16):
        print('{}th challenge.'.format(i))
        c_model = ClassicalIsingModel(j, h, c, neighbor_size=8, state_shape=h.shape, beta=1)
        c_annealer = SimulatedAnnealer(c_model, update_limit=100, freeze_limit=2000)
        c_annealer.anneal(iter_callback=callback)
        energy = c_model.objective_value()
        iter += c_annealer.iter_count
        if energy < min_energy:
            min_energy = energy
            best_annealer = c_annealer

    best_model = best_annealer.model
    print('annealing time: {}'.format(time.time() - start))
    print('annealer: {}'.format(best_annealer))
    print('iterations(average): {}'.format(iter/16))
    print('objective: {}'.format(best_model.objective_value()))
    print('best_state: {}'.format(best_model.state.to_array()))
    print('validity: {}'.format(check_constraints(best_model.state)))

    start = time.time()
    q_model = QuantumIsingModel(j, h, c, neighbor_size=8, state_shape=h.shape, beta=1, gamma=0.1, n_trotter=16)
    q_annealer = QuantumAnnealer(q_model, update_limit=100, freeze_limit=2000)
    q_annealer.anneal(iter_callback=callback)
    observed = q_model.observe_best()
    print('annealing time: {}'.format(time.time() - start))
    print('annealer: {}'.format(q_annealer))
    print('iterations: {}'.format(q_annealer.iter_count))
    print('objective: {}'.format(q_model.objective_value()))
    print('best state: {}'.format(observed.to_array()))
    print('validity: {}'.format(check_constraints(observed)))


def main(argv):
    solve_tsp()


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
