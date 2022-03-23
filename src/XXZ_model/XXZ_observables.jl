###
### XXZ Hamiltonian with nn and nnn interactions.
###

# TODO: Use type parametrisation to generate different hamiltonian functions
# ie use Δ and λ as types and specialise on those.

# TODO: preallocate output vector and reuse it: apply_H!(output, n, L)

# TODO use bits_differ function for periodic boundary condition checks.
"""
Returns the image of n under the action of the XXZ Hamiltonian.
"""
function apply_H(n::Unsigned, L, Δ, λ)

    nn_weight = -1/(1+λ)
    nnn_weight = -λ/(1+λ)
    n_aligned = number_of_aligned_neighbours(n, 1, L)
    diag = (-Δ/2) * (2*n_aligned - L)

    if λ > 0
        n_aligned = number_of_aligned_neighbours(n, 2, L)
        diag2 = (-Δ/2) * (2*n_aligned - L)
        diag = (diag + λ * diag2) / (1+λ)
    end

    output = [(n, diag)]
    for l = 0:L-2
        if bits_differ(n, l, l+1 )
            m = flipbits(n, l, l+1)
            push!(output, (m, nn_weight))
        end
    end

    # Periodic boundary condition
    if (n & 1) ⊻ (n >> (L-1)) != 0
        m = flipbits(n, 0, L-1)
        push!(output, (m, nn_weight))
    end

    if λ > 0
        for l = 0:L-3
            if bits_differ(n, l, l+2)
                m = flipbits(n, l, l+2)
                push!(output, (m, nnn_weight))
            end
        end

        # Periodic boundary condition
        if (n & 1) ⊻ ((n >> (L-2)) & 1) != 0
            m = flipbits(n, 0, L-2)
            push!(output, (m, nnn_weight))
        end
        if (n & 2) ⊻ ((n >> (L-2)) & 2) != 0
            m = flipbits(n, 1, L-1)
            push!(output, (m, nnn_weight))
        end
    end

    output
end


###
### Zero momentum distribution.
###

"""
Returns the image of n under the zero momentum distribution operator.
"""
function apply_A(n::Unsigned, L)
    weight = 1 / L
    diag = hamming_weight(n) / L
    output = [(n, diag)]

    for l in 0:L-2
        for m in l+1:L-1
            if bits_differ(n, l, m)
                o = flipbits(n, l, m)
                push!(output, (o, weight))
            end
        end
    end
    output
end

###
### Nearest neigbour interaction energy density.
###

"""
Returns the image of n under the nearest neigbour interaction energy density operator.
"""
function apply_B(n::Unsigned, L)
    nn = translate(n, L)
    nn = n & nn
    weight = hamming_weight(nn) / L
    [(n, weight)]
end