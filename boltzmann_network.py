# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from typing import List

def neighbours(input_data: np.ndarray, site: int) -> List[np.ndarray]:
    """Return neighbouring spin values alng with their indicies
    [<left>, <right>, <above> , <below>].

    Helliac Periodic Boundary conditions are applied. Asuuming 


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

class BoltzmannNetwork(object):      
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

    def train_weights(self, train_data):
        print("Start to train weights...")
        num_data =  len(train_data)
        # num_data =  len(train_data)
        self.num_neuron = train_data[0].size
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        # this wont work since Lattice2D does not have an iterative method
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for i in range(num_data):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        # self.energy_attract =
        
        self.W = W 
    
    def predict(self, data: List[np.ndarray], num_iter=20, threshold=0, asyn=True):
        # in the orginal code data is a list containing flattened arrays
        # changing it to a list of lattices
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn  # not important for our use
        
        # list aliasing
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        # this for loops all the example images
        # the training happens in the predict and run methods
        # copied_data[i] is a 1xn array
        for i in range(len(data)):
            struct_data = copied_data[i]
            predicted.append(self._run(struct_data))
        return predicted
    
    def _run(self, init_s: np.ndarray):
        """ Asynchronous update

            === Parameters ===
            init_s: n x 1 length array
        """
        # Compute initial state energy
        s = init_s
        e = self.energy(s)
        
        # Iteration
        for i in range(self.num_iter):
            for j in range(15):
                # Select random neuron
                idx = np.random.randint(0, self.num_neuron) 
                # This line is the previous activation function

                # The new way for computing v and s
                s = np.sign(self.W @ s - self.threshold)
                delta_energy, original = self.update_lattice(s, idx) 
            
            # Compute new state energy
            e_new = e + delta_energy
            
            # s is converged
            if np.abs(delta_energy) < 1:
                return s
            # Update energy
            e = e_new
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
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()
