"""
Returns the full matrix representation of the given operator
for the XXZ model of length L.
"""
function build_matrix(operator::Function, L, args...)
    d = 2^L
    H = zeros(d, d)

    for b in UInt32(0):UInt32(2^L-1)
        output = operator(b, L, args...)
        for (a, h) in output
            H[a+1, b+1] += h
        end
    end
    H
end


#======================================================#
# Particle Number Sector
#======================================================#

"""
Returns the matrix representation  the given operator in 
the N-particle sector of the XXZ model of length L.
"""
function build_matrix_N(operator::Function, L, N, args...)
    basis = build_basis_N(UInt32, L, N)
    d = length(basis)
    HN = zeros(d, d)

    index_map = Dict(basis .=> 1:d)

    for (b, n) in enumerate(basis)
        output = operator(n, L, args...)
        for (m, h) in output
            a = index_map[m]
            HN[a, b] += h
        end
    end

    HN
end


#======================================================#
# Momentum Sector
#======================================================#

"""
Returns the matrix representation of the given operator
in the particle-momentum sector N, k for the XXZ model 
with length L.
"""
function build_matrix_Nk(operator::Function, L, N, k, args...)
    basis = build_basis_Nk(UInt32, L, N, k)
    d = length(basis)
    HNk = zeros(d, d)
    index_map = Dict(e => i for (i, (e, p)) in enumerate(basis))
    ωk = cispi(2 * k / L)

    for (b, (n, pn)) in enumerate(basis)
        output = operator(n, L, args...)
        YnL = √pn
        for (m, h) in output
            m_rs, pm, d = representative_state(m, L)

            # Exclude states that don't satisfy the 
            # commensurability condition: k * p divisible by L,
            # as these states end up with zero amplitude.
            if haskey(index_map, m_rs)
                a = index_map[m_rs]
                YmL = (√pm)
                HNk[a, b] += (YnL/YmL) * ωk^d * h |> real
            end
        end
    end

    HNk
end


#======================================================#
# Maximum Symmetry Sector
#======================================================#

"""
Returns the matrix representation for the given XXZ model 
observable in the maximum symmetry sector. L is the length
of the model.
"""
function build_matrix_MSS(operator::Function, L, args...)
    basis = build_MSS_basis(L)
    d = length(basis)
    M = zeros(d, d)
    index_map = Dict(e => i for (i, (e, qp)) in enumerate(basis))

    for (b, (n, qpn)) in enumerate(basis)
        output = operator(n, L, args...)
        Zn4L = √qpn
        for (m, h) in output
            m_srs, qpm = super_representative_state(m, L)
            a = index_map[m_srs]
            Zm4L = √qpm
            M[a, b] += (Zn4L/Zm4L) * h
        end
    end

    M
end

"""
Returns the super representative state of the equivalence classes related to
the equivalence class containing n.
"""
function super_representative_state(n, L)
    n, nx, nr, nrx, p = related_representative_states(n, L)
    n_srs = min(n, nx, nr, nrx)
    n_srs, length(unique((n, nx, nr, nrx))) * p
end