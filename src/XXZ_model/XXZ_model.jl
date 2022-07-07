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

include("basis_utils.jl")
include("XXZ_basis.jl")
include("XXZ_operators.jl")
include("XXZ_sector_matrices.jl")
include("XXZ_time_evolution.jl")
include("state_utils.jl")

# TODO: The following should be separated into a separate module once it's big enough
include("analysis_tools.jl")

end
