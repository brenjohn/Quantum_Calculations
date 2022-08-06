using Polyester

#=============================================================#
# Utility functions for computing entanglement entropy        #
#=============================================================#

"""
Transform a state from the MSS basis to the full basis.
"""
function MSS_to_full_state(ψ, L, basis)
    state = zeros(ComplexF64, 2^L)
    for (amp, (basis_vector, pq)) in zip(ψ, basis)
        full_basis_vectors = XXZ.MSS_to_full_basis(basis_vector, L)
        for n in full_basis_vectors
            state[n+1] += amp * sqrt(pq) / (4*L)
        end
    end
    state
end

"""
Compute the entanglement entropy for the first L1 sites of the given state.
"""
function entanglement_entropy(ψ, L, L1)
    D1 = 2^L1; D2 = 2^(L-L1)
    ψ = reshape(ψ, (D1, D2))
    F = svd(ψ)
    λs = F.S .^ 2
    -sum(x -> x != 0 ? x * log(x) : 0.0, λs)
end

#=============================================================#
# Ensemble Averages                                           #
#=============================================================#

"""
Return the diagonal ensemble average of the operator `O` with respect to
the state `ψ` and the facotrised Hamiltonian `F`.
"""
function diagonal_ensemble_average(O::Matrix, ψ, F)
    Ō = 0.0
    O = F.vectors' * O * F.vectors
    ψ = F.vectors' * ψ
    for n in 1:length(eachcol(F.vectors))
        Ō += abs(ψ[n])^2 * O[n, n]
    end
    Ō
end

diagonal_ensemble_average(Os::Vector{T}, ψ, F) where T <: Matrix = [diagonal_ensemble_average(O, ψ, F) for O in Os]

"""
Return the micro-canonical ensemble average of the operator `O` with respect to
the state `ψ` and the facotrised Hamiltonian `F`.
"""
function micro_canonical_ensemble_average(O::Matrix, ψ, F)
    ψ = F.vectors' * ψ
    prob = abs.(ψ) .^ 2
    E = prob' * F.values
    O = F.vectors' * O * F.vectors
    average, _ = XXZ.coarse_grained_average(diag(O), F.values, [E], 0.02)
    first(average)
end

micro_canonical_ensemble_average(Os::Vector{T}, ψ, F) where T <: Matrix = [micro_canonical_ensemble_average(O, ψ, F) for O in Os]

#=============================================================#
# Expectation value                                           #
#=============================================================#

# TODO: This assumes the N-particle sector is being used
function expectation_value(O, O_args, L, ψ::Dict{U, T}) where U <: Unsigned where T <: Complex

    @batch threadlocal = zero(T)::T for (n, amp) in pairs(ψ)
        for (m, weight) in O(n, L, O_args...)
            threadlocal += ψ[m]' * weight * amp
        end
    end

    sum(threadlocal)
end


#=============================================================#
# Reduced density                                             #
#=============================================================#

"""
Compute the reduced density matrix for the subsystem of the `L`-site
XXZ chain, in state `ψ`, defined by the sites conatined in `sites`.

Trᵦ(ψψ')ᵢⱼ = ∑ₙ <i|<n|ψ ψ'|n>|j> = ∑ᵢ ψ(ni) * ψ(nj)' 
"""
function reduced_density_matrix!(
                            ρ::Matrix{T},
                            ψ::Vector{T}, 
                            sites::Vector{Int}, 
                            L,
                            N
                            ) where T <: Complex
    reduced_basis = decomposed_basis(UInt32, L, N, sites)

    for (basisA, basisB) in reduced_basis
        l = length(basisA)
        @batch for I in CartesianIndices((l, l))
            i, j = Tuple(I)
            mi = basisA[i] 
            mj = basisA[j]
            for n in basisB
                ni = insert_bits(n, mi, sites)
                nj = insert_bits(n, mj, sites)
                ρ[mi+1, mj+1] += ψ[ni+1] * ψ[nj+1]'
            end
        end
    end
    nothing
end

function reduced_density_matrix(ψ::Vector{T}, args...) where T <: Complex
    dim = 2^length(sites)
    ρ = zeros(T, dim, dim)
    reduced_density_matrix!(ρ, ψ, args...)
    ρ
end

"""
Return an array of basis pairs representing `N`-particale basis states of the 
subsystems A and B. A is the subsystem of the `L`-site XXZ chain defined by
the sites contained in `sites` and B is the complement subsystem.

Each element in the array has a pair of basis sets corresponding to a fixed
number of particles in each subsystem.
"""
function decomposed_basis(U::DataType, L, N, sites)
    n = length(sites)
    [(build_basis_N(U, n, i), build_basis_N(U, L-n, N-i)) for i = max(0, N-L+n):min(n, N)]
end

"""
Compute the trace distance between density matrices `ρ` and `σ`.

d(ρ, σ) = 1/2 * Tr[ √(ρ - σ)† * (ρ - σ) ]
"""
function trace_distance(ρ, σ)
    U = ρ - σ
    d = (U' * U) |> sqrt |> diag |> sum
    d / 2
end

function purity(ρ)
    (ρ * ρ) |> diag |> sum
end

function von_Neumann_entropy(ρ)
    d = (ρ * log(ρ)) |> diag
    -sum(d)
end