#!/usr/bin/env python

import collections
import math
import sys

import numpy as np

from anneal.models import IsingModel
from anneal.annealers import SimulatedAnnealer


POSITIONS = [
    (0.0, 0.0),
    (1.0, 2.0),
    (-3.1, 2.1),
    (-0.5, -4.0),
    (0.9, 2.3)
]


def distance(pos1, pos2):
    x = pos1[0] - pos2[0]
    y = pos1[1] - pos2[1]
    return math.sqrt(x**2 + y**2)


def build_j(positions, f=100.0):
    j = collections.defaultdict(float)
    n = len(positions)
    for t in range(n):
        for id1, pos1 in enumerate(positions):
            for id2, pos2 in enumerate(positions):
                if id1 != id2:
                    j[t, id1, (t+1) % n, id2] = distance(pos1, pos2)

    for t in range(n):
        for a in range(n):
            for b in range(n):
                if a == b:
                    break
                j[t, a, t, b] += f

    for t in range(n):
        for u in range(n):
            if t == u:
                break
            for a in range(n):
                j[t, a, u, a] += f

    return j


def build_h(positions, g=100.0):
    n = len(positions)
    h = np.zeros((n, n))

    for t in range(n):
        for a, pos1 in enumerate(positions):
            for b, pos2 in enumerate(positions):
                if a != b:
                    h[t, a] += 2*distance(pos1, pos2)

    for t in range(n):
        for a in range(n):
            h[t, a] += 4*(n-2)*g

    # Enforce spin[0, 0] == 1.
    h[0, 0] -= 10*n*g

    return h


def main(argv):
    j = build_j(POSITIONS)
    h = build_h(POSITIONS)
    n = len(POSITIONS)
    lattice_shape = (n, n)

    model = IsingModel(j=j, h=h, lattice_shape=lattice_shape)
    annealer = SimulatedAnnealer(model)

    energy_history = []

    def callback(annealer, state_is_updated):
        energy_history.append(annealer.model.energy)
        if annealer.iter_count % 10 == 0:
            print('{}''th iter: {}'.format(annealer.iter_count, annealer.model.energy))

    annealer.anneal(iter_callback=callback)
    print('state:')
    print(annealer.model.state)
    route = [np.argmax(_) for _ in annealer.model.state]
    print('Route:')
    print(' '.join([str(_) for _ in route]))
    print('Distance:')
    print(sum(
        distance(POSITIONS[route[i-1]], POSITIONS[route[i]])
        for i in range(len(POSITIONS))
    ))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
