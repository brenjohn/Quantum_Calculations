using LinearAlgebra

#===============================================================#
# Exact time evolution.
#===============================================================#

LocalUpdate{T} = Tuple{Int64, Int64, NTuple{5, T}}

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
# Simple Lie-Trotter-Suzuki time evolution.
#===============================================================#

function apply_ulm!(y::Vector{T}, x::Vector{T}, basis::Vector{U}, l, m, ps) where U <: Unsigned where T <: Complex
    # Diagonal terms
    Threads.@threads for n in basis
        @inbounds xn = x[n+1]
        v = get_occupations(n, l, m)
        @inbounds y[n+1] += xn * ps[v+1]
    end

    # Off-diagonal terms
    Threads.@threads for n in basis
        if bits_differ(n, l, m)
            @inbounds xn = x[n+1]
            o = flipbits(n, l, m)
            @inbounds y[o+1] += xn * ps[5]
        end
    end

    y, x
end

function apply_trotter_step!(y::Vector{T}, x::Vector{T}, basis::Vector{U},
                            trotter_step::Vector{LocalUpdate{T}}) where U <: Unsigned where T <: Complex
    for step in trotter_step
        x, y = apply_ulm!(y, x, basis, step...)
        y .= zero(T)
    end
    x, y
end

function LTS_evolution!(results, ops, x::Vector{T}, L, ϵ, num_steps; 
                        J1 = 1.0, 
                        V1 = 1.0, 
                        J2 = 0.0, 
                        V2 = 0.0,
                        hs = (),
                        is = (), 
                        pbc= true) where T <: Complex
    x = copy(x)
    y = zeros(T, length(x))
    trotter_steps = get_LTS_steps(L, ϵ, J1, V1, J2, V2, hs, is, pbc; elt=T)
    basis = build_basis_N(UInt32, L, L÷2)

    println("\nRunning LTS evolution with $(num_steps) steps")
    for ti in 1:num_steps
        print("\r    Running time step $(ti)")
        for trotter_step in trotter_steps
            x, y = apply_trotter_step!(y, x, basis, trotter_step)
        end
        # f(results, x, ti)
        record_expectation_values!(results, ops, x, ti, basis, L)
    end

    results
end

"""
Lie-Trotter-Suzuki time evolution.
"""
function LTS_evolution!(f, results, x, L, Δ, λ, ϵ, num_steps)
    J1 = -1/(1+λ)
    J2 = -λ/(1+λ)
    V1 = -0.5*Δ/(1+λ)
    V2 = -0.5*λ*Δ/(1+λ)
    kwargs = Dict(:J1 => J1, :J2 => J2, :V1 => V1, :V2 => V2)
    LTS_evolution!(f, results, x, L, ϵ, num_steps; kwargs...)
end

"""
Returns an array of arrays, each containing the parameters for a trotter step.
"""
function get_LTS_steps(L, ϵ, J1, V1, J2, V2, hs, is, pbc; elt::DataType=ComplexF64)
    hs = Dict(is .=> hs); h0 = Dict()
    if isodd(L)
        he = Dict(i => get(hs, i, 0) for i in 0:L-2)
        ho = Dict(L => get(hs, L, 0))
    else
        he = Dict(i => get(hs, i, 0) for i in 0:L-1)
        ho = h0
    end

    trotter_steps = Vector{LocalUpdate{elt}}[]
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

###
### Record expectation value
###

function record_expectation_values!(results, ops, x::Vector{T}, i, basis::Vector{U}, L) where U <: Unsigned where T <: Complex
    for (j, (op, args, kwargs)) in enumerate(ops)
        results[j, i] = get_expectation_value(op, x, basis, L, args...; kwargs...)
    end
    results
end

function get_expectation_value(op, x::Vector{T}, basis::Vector{U}, L, args...; kwargs...) where U <: Unsigned where T <: Complex
    vals = [zero(T) for _ in 1:Threads.nthreads()]
    Threads.@threads for n in basis
        @inbounds amp = x[n+1]
        for (m, weight) in op(n, L, args...; kwargs...)
            @inbounds vals[Threads.threadid()] += x[m+1]' * weight * amp
        end
    end
    sum(vals)
end