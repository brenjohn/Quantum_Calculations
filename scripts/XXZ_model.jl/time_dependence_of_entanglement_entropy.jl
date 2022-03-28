using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie
using Printf

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5; λ = 1.0; L = 20
dt = 0.05; t_range = dt:dt:7
df = DataFrame(t = Float64[], L1s = Vector[], Se = Vector[])

###
### Prepare the initial state and compute its entanglement entropy.
###

basis = XXZ.build_MSS_basis(L)
ψ0 = zeros(length(basis))
ψ0[end] = 1

state = XXZ.MSS_to_full_state(ψ0, L, basis)
entropies = zeros(L+1)
for L1 = 0:L
    entropies[L1+1] = XXZ.entanglement_entropy(state, L, L1)
end
push!(df, (0, 0:L, entropies))

###
### Diagonalise the Hamiltonian and prepare the time evolution operator.
###

H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)
F = Hermitian(H) |> eigen
C = F.vectors' * ψ0
U = cis.(-dt * F.values) |> Diagonal

###
### Evolve the state and compute the entanglement entropy.
###

for i in 1:length(t_range)
    @info "Running for step $i out of $(length(t_range))"
    # Evolve the state forward a single step.
    C = U * C
    ψ = F.vectors * C

    # Compute the entanglement entropy for different partition sizes.
    full_state = XXZ.MSS_to_full_state(ψ, L, basis)
    entropies = zeros(L+1)
    for L1 = 0:L
        entropies[L1+1] = XXZ.entanglement_entropy(full_state, L, L1)
    end
    push!(df, (i*dt, 0:L, entropies))
end

###
### Plot results
###

f = Figure()
a = Axis(f[1, 1], xlabel="L1", ylabel="Sₑ", title="Evolution of Entanglement Entropy")
for d in eachrow(df)
    scatter!(d.L1s, d.Se, label="time = $(d.t)")
end
axislegend(a, position = :rb)

###
### Animate results
###

time_step = Observable(0)
Se = lift(time_step) do step
    df[to_value(step) + 1, :].Se
end
title = @lift @sprintf "Entanglement Entropy at time %.1f" dt * $time_step
framerate = 30

f = Figure(fontsize=35)
a = Axis(f[1, 1], xlabel="L1", ylabel="Sₑ", title=title)
scatter!(0:L, Se)
ylims!(a, 0, 7)

record(f, joinpath(plotsdir(), "entanglement_evolution.gif"), 0:length(t_range);
        framerate = framerate) do t
    time_step[] = t
end