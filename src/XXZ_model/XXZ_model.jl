module XXZ

#=
This module implements functions for building hamiltonian matrices
for the XXZ model in different symmetry sectors. The algorithms
implemented are outlined in Jung-Hoon's 2020 paper.

The module uses the particle/spin basis for the XXZ model as a 
primitive for constructing other basis and Hamiltonians. The basis 
elements are encoded in the binary representation of unsigned 
integers. 
=#

include("utils.jl")
include("XXZ_basis.jl")
include("XXZ_observables.jl")
include("XXZ_sector_matrices.jl")

end
