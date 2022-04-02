#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:55:43 2022

@author: john
"""

from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian, quantum_operator
from quspin.tools.measurements import ent_entropy

from math import factorial
import numpy as np
import matplotlib.pyplot as plt

def my_entanglement_ent(psi, basis, L, L1):
    """
    Returns the entanglement entropy of psi by partitioning that first L1 sites
    """
    Psi = 1j * np.zeros(3**L)
    for state, amp in zip(basis.states, psi):
        Psi[state] = amp
    Psi = np.reshape(Psi, (3**L1, 3**(L-L1)))
    U, S, V = np.linalg.svd(Psi)
    S = S**2
    return -np.sum(S * np.log(S))



##### define model parameters #####
L = 10    # system size
J = 1.0  # xy interaction
h = 1.0  # zz interaction at time t=0
D = 0.1
J3 = 0.1


print("Building System Basis")
basis = spin_basis_1d(L, S='1', Nup=L, pauli=False)


print("Building Hamiltonian")
S_xy = [[1.0 * J / 2.0, i, i+1] for i in range(L-1)]
S_z  = [[h, i] for i in range(L)]
S_zz = [[D, i, i] for i in range(L)]
S_h3 = [[1.0 * J3 / 2.0, i, i+3] for i in range(L-3)]

static = [["+-", S_xy], ["-+", S_xy],
          ["z", S_z], ["zz", S_zz],
          ["+-", S_h3], ["-+", S_h3]]
dynamic = []
H_XY = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)


print("Computing Hamiltonian spectrum")
E, V = H_XY.eigh()


print("Computing Entanglement Entropies")
subsys = range(basis.L // 2) # define subsystem
Sent = np.asarray(
       [my_entanglement_ent(psi, basis, L, 5)
        for psi in V.T]
       )

# TODO: For some reason, using quspin's ent_entropy function gives different
# results.
# subsys = range(basis.L // 2) # define subsystem
# Sent = np.asarray(
#        [ent_entropy(psi, basis, chain_subsys=subsys)["Sent"]
#         for psi in V.T]
#        )



print("Building Scar states")
# First build the full basis for the system.
full_basis = spin_basis_1d(L, S='1', pauli=False)

# Build the J_plus operator
s_plus = [[np.exp(1j * np.pi * i) / 2.0, i, i] for i in range(L)]
S_plus = [["++", s_plus]]
operator_dict = dict(S = S_plus)
no_checks = {'check_herm':False, 'check_symm':False, 'check_pcon':False}
J_plus = quantum_operator(operator_dict, basis=full_basis, **no_checks)

# Use J_plus to generate the scar states
Omega = np.zeros(3**L); Omega[-1] = 1.0
scar_states = [Omega]
for n in range(1, L+1):
    psi = J_plus.matvec(scar_states[-1])
    scar_states.append(psi)
    
# Noramilise the scar states
for n in range(1, L+1):
    N = np.sqrt(factorial(L - n)/(factorial(n) * factorial(L)))
    scar_states[n] *= N
    
# Find the energy and entanglement entropies of the scar states
H_XY_full = hamiltonian(static, dynamic, basis=full_basis, dtype=np.float64)
scar_energies = [H_XY_full.expt_value(psi) for psi in scar_states]
scar_entropies = [my_entanglement_ent(psi, full_basis, L, 5) for psi in scar_states]

#%%
# Plot results
plt.figure(1)
ax = plt.gca()
ax.set_axisbelow(True)
plt.scatter(E, Sent/np.log(2), s=0.5)
plt.scatter(scar_energies, scar_entropies/np.log(2), s=8, color='red')
plt.xlabel("Energy")
plt.ylabel("S / ln(2)")
plt.title("Bipartite entanglement entropy")
plt.grid(linestyle = ":")

plt.plot([-12, 12], (L*np.log(3) - 1)/(2*np.log(2)) * np.ones(2), color='black', linestyle='--')
plt.xlim([-11, 12])

plt.savefig('bipartite_entanglement.png', dpi=300)