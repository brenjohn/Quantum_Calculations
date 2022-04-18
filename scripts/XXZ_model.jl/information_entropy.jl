using DrWatson
using LinearAlgebra
using GLMakie, Makie
using DataFrames
using StatsBase
using LsqFit
using Printf

include(srcdir("XXZ_model/XXZ_model.jl"))

#=

=#

Δ = 1.0
k = 2
h = XXZ.apply_H

S(ψ) = -sum(ci -> abs(ci)^2 * log(abs(ci)^2), ψ)

df = DataFrame(
    L = Float64[],
    df_L = DataFrame[]
    )

λ_range = 0.0:0.01:1.0
L_range = [16, 18, 20]
for L in L_range
    N = L ÷ 3
    df_L = DataFrame(
                λ = Float64[],
                spectrum = Vector{Float64}[],
                entropy = Vector{Float64}[],
                F = Eigen[]
                )
    @info "Collecting results for L = $L"
    H0 = XXZ.build_matrix_Nk(h, L, N, k, Δ, 0.0)
    F0 = Hermitian(H0) |> eigen

    for λ = λ_range
        @info " Running for λ = $λ"
        N = L ÷ 3
        H = XXZ.build_matrix_Nk(h, L, N, k, 1.0, 1.0, λ, λ)
        F = Hermitian(H) |> eigen

        entropies = [F0.vectors' * v |> S for v in eachcol(F.vectors)]

        push!(df_L, (λ, F.values, entropies, F))
    end

    push!(df, (L, df_L))
end


###
### Animate results
###

time_step = Observable(1)
N = length(λ_range)
framerate = 30

title = lift(time_step) do step
    i = to_value(step)
    i = i > N ? 2*N - i + 1 : i
    dfr = df[j, :df_L]
    λ = dfr[i, :λ]
    @sprintf "Information entropy of eigenstates \n H = Hnn + λ Hnnn, N = L/3, k = 2, λ = %.2f" λ
end

f = Figure(fontsize=35, resolution=(1400, 1400))
a = Axis(f[1, 1], xlabel="E/L", ylabel="S / log(0.48 * D)", title=title)
xlims!(a, -0.8, 0.8)
ylims!(a, 0, 1)

plts = []
for j in length(L_range):-1:1
    energies = lift(time_step) do step
        i = to_value(step)
        i = i > N ? 2*N - i + 1 : i
        dfr = df[j, :df_L]
        L = df[j, :L]
        dfr[i, :spectrum] ./ L
    end

    entropies = lift(time_step) do step
        i = to_value(step)
        i = i > N ? 2*N - i + 1 : i
        dfr = df[j, :df_L]
        L = df[j, :L]
        entropies = dfr[i, :entropy]
        entropies ./ log(0.48 * length(entropies))
    end

    plt = scatter!(energies, entropies, markersize=15)
    push!(plts, plt)
end

Legend(f[1, 2], plts, ["L = " * string(l) for l in reverse(L_range)])

record(f, joinpath(plotsdir("XXZ"), "information_entropy_evolution.gif"), 1:2*N;
        framerate = framerate) do t
    time_step[] = t
end