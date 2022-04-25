###
### XXZ Hamiltonian with nn and nnn interactions.
###

# TODO: preallocate output vector and reuse it: apply_H!(output, n, L)

# TODO: use singleton types and multiple dispatch to remove the "if J !=0" lines in hopping and interaction terms.

"""
Returns the image of n under the action of the XXZ Hamiltonian.
"""
function apply_H(n::T, L;
                J1 = 1.0, 
                V1 = 1.0, 
                J2 = 0.0, 
                V2 = 0.0,
                hs = (),
                is = (), 
                pbc= true
                ) where T <: Unsigned

    output = Tuple{T, Float64}[]

    # Apply nearest and next-nearest neighbour hopping.
    J1 != 0 && hopping_term!(output, J1, 1, n, L, pbc)
    J2 != 0 && hopping_term!(output, J2, 2, n, L, pbc)

    # Apply nearest and next-nearest neighbour interaction terms.
    V1 != 0 && neighbour_interaction_term!(output, V1, 1, n, L, pbc)
    V2 != 0 && neighbour_interaction_term!(output, V2, 2, n, L, pbc)

    # Apply sigma-z impurities
    for (h, i) in zip(hs, is)
        h != 0 && single_site_impurity!(output, h, i, n)
    end

    output
end

function apply_H(n::Unsigned, L, Δ, λ)
    J1 = -1/(1+λ)
    J2 = -λ/(1+λ)
    V1 = Δ/(1+λ)
    V2 = λ*Δ/(1+λ)
    apply_H(n, L; J1=J1, V1=V1, J2=J2, V2=V2)
end

"""
Generate the action of `d`-nearest neighbour hopping, with strength `J`, on 
the state `n` of a system of length `L` and push it to the `output` vector.
"""
function hopping_term!(output, J, d, n, L, pbc)
    for l = 0:L-1-d
        if bits_differ(n, l, l+d)
            m = flipbits(n, l, l+d)
            push!(output, (m, J))
        end
    end

    # Periodic boundary condition
    if pbc
        for b = 0:d-1
            if bits_differ(n, b, L-d+b)
                m = flipbits(n, b, L-d+b)
                push!(output, (m, J))
            end
        end
    end
    nothing
end

"""
Generate the action of `d`-nearest neighbour interaction, with strength `V`, on 
the state `n` of a system of length `L` and push it to the `output` vector.
"""
function neighbour_interaction_term!(output, V, d, n, L, pbc)
    n_aligned = number_of_aligned_neighbours2(n, d, L, pbc)
    diag = (-V/2) * (2*n_aligned - L)
    push!(output, (n, diag))
    nothing
end

function single_site_impurity!(output, h, i, n)
    ni = (n >> i) & 1
    s = 2 * ni - 1
    push!(output, (n, s * h))
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

###
### Local time evolution operator.
###

function apply_ulm!(a::Vector{<:Complex}, ψ::Vector{<:Complex}, l, m, L, Δ, ϵ)
    for n in UInt32(0):UInt32(2^L-1)
        if bits_differ(n, l, m)
            a[n+1] += ψ[n+1] * cis(-Δ * ϵ / 2) * cos(ϵ)
            o = flipbits(n, l, m) # here it is assumed l < m
            a[o+1] += ψ[n+1] * cis(-Δ * ϵ / 2) * sin(ϵ) * 1im
        else
            a[n+1] += ψ[n+1] * cis(Δ * ϵ / 2)
        end
    end
    a, ψ
end

function apply_ulm!(a::Vector{<:Complex}, ψ::Vector{<:Complex}, sites::Vector{<:NTuple{2, Integer}}, L, Δ, ϵ)
    for (l, m) in sites
        ψ, a = apply_ulm!(a, ψ, l, m, L, Δ, ϵ)
        for i = 1:length(a) a[i] = 0 end
    end
    ψ, a
end

"""
L is assumed to be even,
"""
function LTS_evolution(ψ0, L, Δ, λ, ϵ, N_steps)
    if λ == 0
        return _LTS_evolution_nn_interactions(ψ0, L, Δ, ϵ, N_steps)
    elseif L % 4 == 0
        return _LTS_evolution_nnn_interactions(ψ0, L, Δ, λ, ϵ, N_steps)
    else
        error("nnn LTS evolution is only supported when L is a multiple of 4.")
    end
end

function _LTS_evolution_nn_interactions(ψ0::Vector{T}, L, Δ, ϵ, N_steps) where T <: Complex
    ψt = zeros(T, length(ψ0), N_steps)
    ψt[:, 1] = ψ0
    ψ = copy(ψ0)
    a = zeros(T, length(ψ))

    h0_sites = [(2l    , 2l + 1) for l = 0:(L÷2)-1]
    h1_sites = [(2l + 1, 2l + 2) for l = 0:(L÷2)-1]; h1_sites[end] = (0, L-1)

    for ti in 1:N_steps-1
        ψ, a = XXZ.apply_ulm!(a, ψ, h0_sites, L, Δ, ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h1_sites, L, Δ, ϵ)
        ψ, a = XXZ.apply_ulm!(a, ψ, h0_sites, L, Δ, ϵ/2)
        ψt[:, ti+1] = ψ[:]
    end

    ψt
end

function _LTS_evolution_nnn_interactions(ψ0::Vector{T}, L, Δ, λ, ϵ, N_steps) where T <: Complex
    ψt = zeros(T, length(ψ0), N_steps)
    ψt[:, 1] = ψ0
    ψ = ψ0
    a = zeros(T, length(ψ))

    nn_weight  = 1/(1+λ)
    nnn_weight = λ/(1+λ)

    h0_sites = [(2l    , 2l + 1) for l = 0:(L÷2)-1]
    h1_sites = [(2l + 1, 2l + 2) for l = 0:(L÷2)-1]; h1_sites[end] = (0, L-1)

    h2_sites = [(4l    , 4l + 2) for l = 0:(L÷4)-1]
    h2_sites = vcat(h2_sites, [(4l + 1, 4l + 3) for l = 0:(L÷4)-1])

    h3_sites = [(4l + 2, 4l + 4) for l = 0:(L÷4)-1]; h3_sites[end] = (0, L-2)
    h3_sites = vcat(h3_sites, [(4l + 3, 4l + 5) for l = 0:(L÷4)-1]); h3_sites[end] = (1, L-1)

    for ti in 1:N_steps-1
        ψ, a = XXZ.apply_ulm!(a, ψ, h0_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h1_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h2_sites, L, Δ, nnn_weight * ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h3_sites, L, Δ, nnn_weight * ϵ)
        ψ, a = XXZ.apply_ulm!(a, ψ, h2_sites, L, Δ, nnn_weight * ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h1_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = XXZ.apply_ulm!(a, ψ, h0_sites, L, Δ, nn_weight * ϵ/2)

        ψt[:, ti+1] = ψ[:]
    end

    ψt
end