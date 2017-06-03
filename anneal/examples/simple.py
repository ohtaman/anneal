# -*- coding:utf-8 -*-

import sys
import time

import numpy as np
import scipy.sparse as sp

from anneal.models import ClassicalIsingModel, QuantumIsingModel
from anneal.annealers import SimulatedAnnealer, QuantumAnnealer


def callback(annealer, state_is_updated, model_is_updated):
    if annealer.iter_count % 50 == 0:
        print("{}: {}'th iter.".format(
            annealer.__class__.__name__,
            annealer.iter_count
        ))


def main(argv):
    j = {(0, 1): 1}
    h = [0, 0]

    classical_satrt = time.time()
    classical_model = ClassicalIsingModel(j, h, state_size=2, beta=10)
    classical_annealer = SimulatedAnnealer(classical_model)
    classical_annealer.anneal(iter_callback=callback)
    classical_time = time.time() - classical_satrt

    quantum_start = time.time()
    quantum_model = QuantumIsingModel(j, h, state_size=2, beta=10, n_trotter=16)
    quantum_annealer = QuantumAnnealer(quantum_model)
    quantum_annealer.anneal(iter_callback=callback)
    quantum_time = time.time() - quantum_start

    print("SimulatedAnnealer annealing time: {}".format(classical_time))
    print("SimulatedAnnealer iter_count: {}".format(classical_annealer.iter_count))
    print("SimulatedAnnealer objective: {}".format(classical_model.energy()))
    print("QuantumAnnealer annealing time: {}".format(quantum_time))
    print("QuantumAnnealer iter_count: {}".format(quantum_annealer.iter_count))
    print("QuantumAnnealer objective: {}".format(quantum_model.objective_value()))
    print("QuantumAnnealer energy: {}".format(quantum_model.energy()))


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
