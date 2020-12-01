# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from typing import List
import scipy.spatial as ss
from network import HopfieldNetwork

ERROR = 1e-5

def neighbours(input_data: np.ndarray, site: int) -> List[np.ndarray]:
    """Return neighbouring spin values along with their indices
    [<left>, <right>, <above> , <below>].

    Helliac Periodic Boundary conditions are applied. Assuming 


    === Parameters ===
    input_data: a n x 1 array
    """
    size = input_data.size
    L = int(np.sqrt(size))
    left = (site - 1) % size
    right = (site + 1) % size
    up = (site + L) % size
    down = (site - L) % size

    left_ = input_data[left]
    right_ = input_data[right]
    up_ = input_data[up]
    down_ = input_data[down]

    values = np.array([left_, right_, up_, down_])
    idx = np.array([left, right, up, down])

    return [values, idx]

class BoltzmannNetwork(HopfieldNetwork):      
    """ Run a Hopfield Neural Network

    ==== Attributes ===
        num_neron: the size of the flattened training data
        W: the weight matrix
        num_iter: the maximum number of iterations
        threshold: The convergence threshold
        asyn: ?
        exp1/2: Precomputed exponent values
    """
    num_neuron: int
    W: np.ndarray
    num_inter: int
    threshold: float
    asyn: bool
    exp1: np.ndarray
    exp1: np.ndarray

    def _run(self, init_s: np.ndarray):
        """ Asynchronous update

            === Parameters ===
            init_s: n x 1 length array
        """
        # Compute initial state energy
        s = init_s
        e = self.energy(s)
        counter = 0
        
        # Iteration
        for i in range(self.num_iter):
            counter += 1
            original = np.copy(s)
            for j in range(15):
                # Select random neuron
                idx = np.random.randint(0, self.num_neuron) 
                s = np.sign(self.W @ s - self.threshold)
                delta_energy, _ = self.update_lattice(s, idx) 
            
            # Compute new state energy
            e_new = e + delta_energy
            
            # s is converged
            if ss.distance.hamming(original, s) > 0.95:
                self.count.append(counter)
                return s
            # Update energy
            e = e_new
        self.count.append(counter)
        return s
    
    def _acceptance(self, original: np.ndarray, site: int):
        """Return the acceptance probability according to the metropolis
        algorithm w/ single spin dynamic.

        ==== Parameters ====
        original: 1 x n np.ndarray
        site: location of data point in lattice [0, size ** 2 -1]"""
        # this looks weird for the delta energy term but check page 52 of the
        # monte carlo documents i linked as well as my personal notes in the
        # drive
        nearest_neigh, nn_idx = neighbours(original, site)
        w_ik = self.W[nn_idx, site]
        assert(w_ik.size == 4)
        delta_energy = original[site] * ( np.dot(w_ik, nearest_neigh) - 2)

        return np.exp(-1 * delta_energy/1.5), delta_energy

    def update_lattice(self, original: np.ndarray, site: int):
        """ update the lattice according to the acceptance probability

        ==== Parameters ====
        original: 1 x n np.ndarray
        site: location of data point in lattice [0, size ** 2 -1]"""
        number = np.random.random_sample()

        accept = self._acceptance(original, site)

        if number < accept[0]:
            original[site] = -1 * original[site]

        return accept[1], original
    
