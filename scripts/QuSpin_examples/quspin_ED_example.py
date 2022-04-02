#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:20:27 2022

@author: john
"""

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import matplotlib.pyplot as plt

##### define model parameters #####
L     = 12               # system size
Jxy   = np.sqrt(2.0)     # xy interaction
Jzz_0 = 1.0              # zz interaction
hz    = 1.0/np.sqrt(3.0) # z external field

##### set up Heisenberg Hamiltonian in an enternal z-field #####
# compute spin-1/2 basis
basis   = spin_basis_1d(L, pauli=False)
basisN  = spin_basis_1d(L, pauli=False, Nup=L//2) # zero magnetisation sector
basisNp = spin_basis_1d(L, pauli=False, Nup=L//2, pblock=1) # positive parity

# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,   i, i+1] for i in range(L-1)]
J_xy = [[Jxy/2.0, i, i+1] for i in range(L-1)]
h_z  = [[hz,i] for i in range(L-1)]

# static and dynamic lists
static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz]]
dynamic=[]

# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

##### various exact diagonalisation routines #####
# E = H_XXZ.eigvalsh() # calculate entire spectrum only
E, V = H_XXZ.eigh()   # calculate full eigensystem

# calculate the eigenstate closest to energy E_star
E_star = 0.0
E, psi_0 = H_XXZ.eigsh(k=1, sigma=E_star, maxiter=1E4)
psi_0 = psi_0.reshape((-1,))

plt.plot(E)