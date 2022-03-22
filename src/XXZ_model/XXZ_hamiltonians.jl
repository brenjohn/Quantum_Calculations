#======================================================#
# Hamiltonian
#======================================================#

# TODO: Use type parametrisation to generate different hamiltonian functions
# ie use Δ and λ as types and specialise on those.

# TODO: preallocate output vector and reuse it: apply_H!(output, n, L)

# TODO use bits_differ function for periodic boundary condition checks.
"""
Returns the image of n under the action of the XXZ Hamiltonian.
"""
function apply_H(n::Unsigned, L, Δ, λ)

    n_aligned = number_of_aligned_neighbours(n, 1, L)
    diag = (-Δ/2) * (2*n_aligned - L)

    if λ > 0
        n_aligned = number_of_aligned_neighbours(n, 2, L)
        diag2 = (-Δ/2) * (2*n_aligned - L)
        diag = (diag + λ * diag2) / (1+λ)
    end

    output = [(n, diag)]
    for l = 0:L-2
        if bits_differ(n, l, l+1 )# ((n & (1 << l)) << 1) ⊻ (n & (1 << (l+1))) != 0 # If n[l] != n[l+1]
            m = flipbits(n, l, l+1)
            push!(output, (m, -1/(1+λ)))
        end
    end

    # Periodic boundary condition
    if (n & 1) ⊻ (n >> (L-1)) != 0
        m = flipbits(n, 0, L-1)
        push!(output, (m, -1/(1+λ)))
    end

    if λ > 0
        for l = 0:L-3
            if bits_differ(n, l, l+2) #((n & (1 << l)) << 2) ⊻ (n & (1 << (l+2))) != 0 # If n[l] != n[l+2]
                m = flipbits(n, l, l+2)
                push!(output, (m, -λ/(1+λ)))
            end
        end

        # Periodic boundary condition
        if (n & 1) ⊻ ((n >> (L-2)) & 1) != 0
            m = flipbits(n, 0, L-2)
            push!(output, (m, -λ/(1+λ)))
        end
        if (n & 2) ⊻ ((n >> (L-2)) & 2) != 0
            m = flipbits(n, 1, L-1)
            push!(output, (m, -λ/(1+λ)))
        end
    end

    output
end

function build_H(L, Δ = 2, λ = 0.2)
    d = 2^L
    H = zeros(d, d)

    for b in UInt32(0):UInt32(2^L-1)
        output = apply_H(b, L, Δ, λ)
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
Returns the Hamiltonian for the N-particle sector of the XXZ model
of length L.
"""
function build_HN(L, N, Δ = 2, λ = 0.2)
    basis = build_basis_N(UInt32, L, N)
    d = length(basis)
    HN = zeros(d, d)

    index_map = Dict(basis .=> 1:d)

    for (b, n) in enumerate(basis)
        output = apply_H(n, L, Δ, λ)
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
Returns the Hamiltonian of the particle-momentum sector N, k 
for the XXZ model with length L.
"""
function build_HNk(L, N, k, Δ = 2, λ = 0.2)
    basis = build_basis_Nk(UInt32, L, N, k)
    d = length(basis)
    HNk = zeros(d, d)
    index_map = Dict(e => i for (i, (e, p)) in enumerate(basis))
    ωk = cispi(2 * k / L)

    for (b, (n, pn)) in enumerate(basis)
        output = apply_H(n, L, Δ, λ)
        YnL = √pn
        for (m, h) in output
            m_rs, pm, d = representative_state(m, L)
            a = index_map[m_rs]
            YmL = (√pm)
            HNk[a, b] += (YnL/YmL) * ωk^d * h
        end
    end

    HNk
end


#======================================================#
# Maximum Symmetry Sector
#======================================================#

"""
Returns the Hamiltonian for the XXZ model of length L in the
maximum symmetry sector.
"""
function build_HMSS(L, Δ = 2, λ = 0.2)
    basis = build_MSS_basis(L)
    d = length(basis)
    HMSS = zeros(d, d)
    index_map = Dict(e => i for (i, (e, qp)) in enumerate(basis))

    for (b, (n, qpn)) in enumerate(basis)
        output = apply_H(n, L, Δ, λ)
        Zn4L = √qpn
        for (m, h) in output
            m_srs, qpm = super_representative_state(m, L)
            a = index_map[m_srs]
            Zm4L = √qpm
            HMSS[a, b] += (Zn4L/Zm4L) * h
        end
    end

    HMSS
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