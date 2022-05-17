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
    value = zero(T)

    for (n, amp) in pairs(ψ)
        for (m, weight) in O(n, L, O_args...)
            value += ψ[m]' * weight * amp
        end
    end

    value
end