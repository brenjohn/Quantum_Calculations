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