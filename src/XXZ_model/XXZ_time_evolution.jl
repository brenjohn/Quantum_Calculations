using LinearAlgebra

#===============================================================#
# Exact time evolution.
#===============================================================#

"""
Evolves the initial state `ψᵢ` over the time range `t_range` according to the
factorised hamiltonian `F`.

Returns a matrix whose i-th coloumn is the evolved state at time tᵢ.
"""
function time_evolve(ψᵢ, F, dt, steps)
    Ct = zeros(ComplexF64, length(ψᵢ), steps + 1)
    Ct[:, 1] = F.vectors' * ψᵢ

    U = cis.(-dt * F.values) |> Diagonal
    for i in 1:steps
        Ct[:, i+1] = U * Ct[:, i]
    end

    F.vectors * Ct
end


#===============================================================#
# Lie-Trotter-Suzuki time evolution.
#===============================================================#

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
Lie-Trotter-Suzuki time evolution.

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
        ψ, a = apply_ulm!(a, ψ, h0_sites, L, Δ, ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h1_sites, L, Δ, ϵ)
        ψ, a = apply_ulm!(a, ψ, h0_sites, L, Δ, ϵ/2)
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
        ψ, a = apply_ulm!(a, ψ, h0_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h1_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h2_sites, L, Δ, nnn_weight * ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h3_sites, L, Δ, nnn_weight * ϵ)
        ψ, a = apply_ulm!(a, ψ, h2_sites, L, Δ, nnn_weight * ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h1_sites, L, Δ, nn_weight * ϵ/2)
        ψ, a = apply_ulm!(a, ψ, h0_sites, L, Δ, nn_weight * ϵ/2)

        ψt[:, ti+1] = ψ[:]
    end

    ψt
end






####################################################################

"""
"""
function LTS_evolution(O, O_args, ψ0::Vector{T}, basis::Vector{U}, L, ϵ, num_steps; 
                    J1 = 1.0, 
                    V1 = 1.0, 
                    J2 = 0.0, 
                    V2 = 0.0,
                    hs = (),
                    is = (), 
                    pbc= true) where T <: Complex where U <: Unsigned
    x = Dict{U, T}(basis .=> ψ0)
    y = Dict{U, T}(basis .=> zero(T))
    results = zeros(T, num_steps + 1)

    trotter_steps = get_LTS_steps(L, ϵ, J1, V1, J2, V2, hs, is, pbc; elt=T)

    for ti in 1:num_steps
        for trotter_step in trotter_steps
            x, y = apply_ulm!(y, x, basis, trotter_step)
        end

        # TODO: record result
        results[ti + 1] = expectation_value(O, O_args, L, x)
    end
    # return results
    results
end

function get_LTS_steps(L, ϵ, J1, V1, J2, V2, hs, is, pbc; elt::DataType=ComplexF64)
    hs = Dict(is .=> hs); h0 = Dict()
    if isodd(L)
        he = Dict(i => get(hs, i, 0) for i in 0:L-2)
        ho = Dict(L => get(hs, L, 0))
    else
        he = Dict(i => get(hs, i, 0) for i in 0:L-1)
        ho = h0
    end

    trotter_steps = []
    if J2 == 0 && V2 == 0
        push!(trotter_steps, [ulm_elements(elt, ϵ/2, J1, V1, he, sites...) for sites in trotter_sites(0, 1, L, pbc)])
        push!(trotter_steps, [ulm_elements(elt, ϵ  , J1, V1, ho, sites...) for sites in trotter_sites(1, 1, L, pbc)])
    else
        push!(trotter_steps, [ulm_elements(elt, ϵ/2, J1, V1, he, sites...) for sites in trotter_sites(0, 1, L, pbc)])
        push!(trotter_steps, [ulm_elements(elt, ϵ/2, J1, V1, ho, sites...) for sites in trotter_sites(1, 1, L, pbc)])
        push!(trotter_steps, [ulm_elements(elt, ϵ/2, J2, V2, h0, sites...) for sites in trotter_sites(0, 2, L, pbc)])
        push!(trotter_steps, [ulm_elements(elt, ϵ,   J2, V2, h0, sites...) for sites in trotter_sites(1, 2, L, pbc)])
    end
    [trotter_steps; trotter_steps[end-1:-1:1]]
end

"""
Returns the `nn`-th nearest neighbour site pairs participating in trotter step `n`.

Returns an incorrect result when both `L` is odd and `pbc` is true.
"""
function trotter_sites(n, nn, L, pbc)
    if isodd(L) && pbc
        # Not supported because of the extra trotter step required for this case.
        error("Periodic boundary conditions is currently not supported for odd length chains")
    end

    s = p -> p[1] < p[2] ? p : (p[2], p[1])
    p = (l, i) -> ((l+nn*n+i)%L, (l+nn*(n+1)+i)%L)
    pairs = [p(l, i) |> s for l in 0:2*nn:L-1 for i in 0:nn-1 if pbc || l+nn*(n+1)+i < L]
    unique(p -> p[1], pairs) # This is only required if nn > 2.
end

"""
⌈exp(iV)           ⌉⌈exp(i(h₁+h₂))                                                            ⌉
∣   exp(-iV)       ∣∣             cos(λ) + a/λ i sin(λ)    J/λ i sin(λ)                       ∣
∣       exp(-iV)   ∣∣             J/λ i sin(λ)             cos(λ) - a/λ i sin(λ)              ∣
⌊           exp(iV)⌋⌊                                                           exp(-i(h₁+h₂))⌋

where a = (h₁-h₂), λ = √(a²+J²). (Assumes l < m)
"""
function ulm_elements(::Type{T}, ϵ, J, V, hs, l, m) where T <: Complex
    h1, h2 = get(hs, l, 0), get(hs, m, 0)
    λ = (h1 - h2)^2 + J^2 |> sqrt
    a = cis(-ϵ * (V + h1 + h2))
    b = cis(ϵ * V) * (cos(-ϵ * λ) + 1im * (h1 - h2)/λ * sin(-ϵ * λ))
    c = cis(ϵ * V) * (cos(-ϵ * λ) - 1im * (h1 - h2)/λ * sin(-ϵ * λ))
    d = cis(-ϵ * (V - h1 - h2))
    e = cis(ϵ * V) * 1im * J/λ * sin(-ϵ * λ)
    (l, m, convert.(T, (a, b, c, d, e)))
end

function apply_ulm!(y::Dict{U, T}, x::Dict{U, T}, basis, sites) where U <: Unsigned where T <: Complex
    for (l, m, ps) in sites
        x, y = apply_ulm!(y, x, basis, l, m, ps)
        y.vals .= zero(T) # TODO: This is a bit hacky, but avoids computing hashes.
    end
    x, y
end

"""
    apply_ulm!(y::Vector{T}, x::Vector{T}, l, m, ps::NTuple{4, T}) where T <: Complex

Writes to `y` the image of state `x` under the two site operator uₗₘ, defined as:

       ⌈a          ⌉
 uₗₘ = ∣   b   e   ∣
       ∣   e   c   ∣
       ⌊          d⌋

acting on sites l and m with ps = (a, b, c, d, e).
"""
function apply_ulm!(y::Dict{U, T}, x::Dict{U, T}, 
                    basis::Vector{U}, l, m, ps::NTuple{5, T}) where U <: Unsigned where T <: Complex
    for n in basis
        V = get_occupations(n, l, m) |> Val
        ulm_update!(V, y, n, x[n], l, m, ps)
    end
    y, x
end

ulm_update!(::Val{0}, y::Dict{U, T}, n::U, x::T, l, m, ps::NTuple{5, T}) where U <: Unsigned where T <: Complex = y[n] += x * ps[1]
ulm_update!(::Val{3}, y::Dict{U, T}, n::U, x::T, l, m, ps::NTuple{5, T}) where U <: Unsigned where T <: Complex = y[n] += x * ps[4]

function ulm_update!(::Val{1}, y::Dict{U, T}, n::U, x::T, l, m, ps::NTuple{5, T}) where U <: Unsigned where T <: Complex
    y[n] += x * ps[2]
    o = flipbits(n, l, m) # here it is assumed l < m
    y[o] += x * ps[5]
end

function ulm_update!(::Val{2}, y::Dict{U, T}, n::U, x::T, l, m, ps::NTuple{5, T}) where U <: Unsigned where T <: Complex
    y[n] += x * ps[3]
    o = flipbits(n, l, m) # here it is assumed l < m
    y[o] += x * ps[5]
end