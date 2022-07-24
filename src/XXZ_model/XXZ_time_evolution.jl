using LinearAlgebra
using Polyester

# TODO: Avoid storing the full state vector in LTS evolution to make it more memory efficient.
# TODO: The magnetisation sector should be picked by the user for LTS evolution.
# TODO: The LTS implementation assumes a particular magnetisation sector. This should be changed by
#       adding the basis to be used as an input argument.


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
#
# For an overview of this method see the Jung-Hoon 2020 tutorial.
#
# The solution to the Schrodinger equation:
#
#     ψ(t) = exp(-it H) * ψ(0) = exp(-idt H)^n * ψ(0), (t/n = dt)
#
# is computed by decomposing the Hamiltonian into sums of mutually
# commuting terms and using Baker–Campbell–Hausdorff formula to
# write
#
#     exp(idtH) ≈ exp(idt H₀) * exp(idt H₁) ...
#
# Each factor on the right hand side defines a trotter step. Each
# trotter step can then be further factorised using the fact the
# associated hamiltonian is a sum of commuting terms Hᵢ = ∑hₗₘ :
#
#    exp(idt Hᵢ) = Πₗₘ exp(idt hₗₘ) = Πₗₘ Uₗₘ
#
# Each factor of the right hand side defines a local update step.
# In the implementation below, the local update step is 
# multi-threaded.
#===============================================================#

# A useful type alias
LocalUpdate{T} = Tuple{Int64, Int64, NTuple{5, T}}

"""
Time evolve a state `x` of the XXZ model using Lie-Trotter-Suzuki 
time evolution. After each time step, the expecatation values of
operators in the `ops` argument are calculated and stored in the
`results` argument.

This implementation is multi-threaded.

# arguments
- `results`: A variable to store expectation values.
- `ops`: An array of operators to record expectation values for.
- `x`: The initial state to be eolved.
- `L`: The length of the XXZ chain.
- `ϵ`: The time step to be used for the evolution.
- `num_steps`: The number of time steps to take.

# keyword arguments
- `J1`: The strength of nearest neighbour hopping.
- `V1`: The strength of nearest neighbour interaction.
- `J2`: The strength of next nearest neighbour hopping.
- `V2`: The strength of next nearest neighbour interaction.
- `hs`: An iterable containing the single site impurity strengths.
- `is`: An iterable containing the locations of single site impurities.
- `pbs`: `true` if periodic boundary conditions are to be imposed.
"""
function LTS_evolution!(results, ops, x::Vector{T}, L, ϵ, num_steps; 
                        J1 = 1.0, 
                        V1 = 1.0, 
                        J2 = 0.0, 
                        V2 = 0.0,
                        hs = (),
                        is = (), 
                        pbc= true) where T <: Complex
    # Set up variables for the time evolution.
    ti = time()
    x = copy(x)
    y = zeros(T, length(x))
    trotter_steps = get_LTS_steps(L, ϵ, J1, V1, J2, V2, hs, is, pbc; elt=T)
    basis = build_basis_N(UInt32, L, L÷2) # TODO: the magnetisation sector should be picked by the user.

    # Perform the LTS evolution and record results.
    println("\nRunning LTS evolution with $(num_steps) steps")
    for ti in 1:num_steps
        # print("\r    Running time step $(ti)")
        for trotter_step in trotter_steps
            x, y = apply_trotter_step!(y, x, basis, trotter_step)
        end
        record_expectation_values!(results, ops, x, ti, basis, L)
    end

    println("\nTime taken by LTS evolution: $(time() - ti)\n")
    results
end

"""
Use Lie-Trotter-Suzuki time evolution with the parameterisation
of the Hamiltonian used in the Jung-Hoon 2020 tutorial.
"""
function LTS_evolution!(results, ops, x, L, Δ, λ, ϵ, num_steps)
    J1 = -1/(1+λ)
    J2 = -λ/(1+λ)
    V1 = -0.5*Δ/(1+λ)
    V2 = -0.5*λ*Δ/(1+λ)
    kwargs = Dict(:J1 => J1, :J2 => J2, :V1 => V1, :V2 => V2)
    LTS_evolution!(results, ops, x, L, ϵ, num_steps; kwargs...)
end

"""
Apply all local update steps contained in the given trotter step to the given state x.
"""
function apply_trotter_step!(y::Vector{T}, x::Vector{T}, basis::Vector{U},
                            trotter_step::Vector{LocalUpdate{T}}) where U <: Unsigned where T <: Complex
    for ulm in trotter_step
        x, y = apply_ulm!(y, x, basis, ulm...)
        y .= zero(T) # clear the y variable for the next local update.
    end
    x, y
end

"""
Apply the given local update to the state `x` and store the result in `y`.
"""
function apply_ulm!(y::Vector{T}, x::Vector{T}, basis::Vector{U}, l, m, ps) where U <: Unsigned where T <: Complex
    # Note, the following for loops are not fused for thread safety.
    # Diagonal terms
    @batch for n in basis
        @inbounds xn = x[n+1]
        v = get_occupations(n, l, m)
        @inbounds y[n+1] += xn * ps[v+1]
    end

    # Off-diagonal terms
    @batch for n in basis
        if bits_differ(n, l, m)
            @inbounds xn = x[n+1]
            o = flipbits(n, l, m)
            @inbounds y[o+1] += xn * ps[5]
        end
    end

    y, x
end



#===============================================================#
# Functions for setting up a LTS evolution.
#===============================================================#

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
Returns a tuple containing all the parameters needed to perform the local update corresponding
to sies `l` and `m` with hamiltonian parameters `J`, `V` and `hs`.

The third element of the returned tuple contains the matrix elements of the local update
(a, b, c, d, e) and come from the following: 

        ⌈a         ⌉
uₗₘ  =  ∣   b  e   ∣  =
        ∣   e  c   ∣
        ⌊         d⌋

⌈exp(iV)           ⌉⌈exp(i(h₁+h₂))                                                            ⌉
∣   exp(-iV)       ∣∣             cos(λ) + α/λ i sin(λ)    J/λ i sin(λ)                       ∣
∣       exp(-iV)   ∣∣             J/λ i sin(λ)             cos(λ) - α/λ i sin(λ)              ∣
⌊           exp(iV)⌋⌊                                                           exp(-i(h₁+h₂))⌋

where α = (h₁-h₂), λ = √(a²+J²). (Note, this assumes l < m)
"""
function ulm_elements(::Type{T}, ϵ, J, V, hs, l, m) where T <: Complex
    # The extra -ϵ that appears below comes from exp(-iϵH) in the solution
    # to the schrodinger equation.
    h1, h2 = get(hs, l, 0), get(hs, m, 0)
    λ = (h1 - h2)^2 + J^2 |> sqrt
    a = cis(-ϵ * (V + h1 + h2))
    b = cis(ϵ * V) * (cos(-ϵ * λ) + 1im * (h1 - h2)/λ * sin(-ϵ * λ))
    c = cis(ϵ * V) * (cos(-ϵ * λ) - 1im * (h1 - h2)/λ * sin(-ϵ * λ))
    d = cis(-ϵ * (V - h1 - h2))
    e = cis(ϵ * V) * 1im * J/λ * sin(-ϵ * λ)
    (l, m, convert.(T, (a, b, c, d, e)))
end



#===============================================================#
# Functions for recording expectation values.
#===============================================================#

"""
Compute the expectation value of the given operators and record them in the `results` variable.
"""
function record_expectation_values!(results, ops, x::Vector{T}, i, basis::Vector{U}, L) where U <: Unsigned where T <: Complex
    for (j, (op, args, kwargs)) in enumerate(ops)
        results[j, i] = get_expectation_value(op, x, basis, L, args...; kwargs...)
    end
    results
end

"""
Compute the expectation value of the given operator `op` with respect to the given state vector `x`.
"""
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