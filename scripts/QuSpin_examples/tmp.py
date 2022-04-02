#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:33:55 2022

@author: john
"""

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian, quantum_operator
from quspin.tools.measurements import ent_entropy

import numpy as np
import matplotlib.pyplot as plt

##### define model parameters #####
L = 10

print("Building System Basis")
basis = spin_basis_1d(L, S='1', pauli=True)

print("Building Hamiltonian")
S_z  = [[1.0, i, i] for i in range(L)]

static = [["zz", S_z]]
dynamic = []

H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64, check_herm=False, check_symm=False, check_pcon=False)

# a = 1.0
# static = [["+", S_z]]
# input_dict = {"a" : static}
# O = quantum_operator(input_dict, L)


print("Building Hamiltonian")
j_plus  = [[np.exp(1j * np.pi * i) / 2.0, i, i] for i in range(L)]
J_plus = [["++", j_plus]]

operator_dict = dict(J = J_plus)
O = quantum_operator(operator_dict, basis=basis, check_herm=False, check_symm=False, check_pcon=False)