using DrWatson
using LinearAlgebra
using GLMakie, Makie
using DataFrames
using StatsBase
using Printf

include(srcdir("XXZ_model/XXZ_model.jl"))

#=
This script computes the level statistics for the XXZ chain with an impurity
at a single site for different strengths of the impurity. 
(ie the hamiltonian is H = Hxxz + h * σzi where i is a specific site and h varies.)

The model is integrable when h = 0 and non-integrable for non-zero h. Thus the level
statistics should change from Poisson to Wigner-Dyson as h increases.
=#


# Set parameters for the computation.
L = 12; N = L ÷ 2
J1 = 1.0; V1 = 1.0; pbc = false; is = (N,)

# Collect and process spectra for different values of h.
h = XXZ.apply_H
bins = 0.0:0.1:4.0
h_range = 0.0:0.01:1.0
df_h = DataFrame(
                h = Float64[],
                unfolded_spec = Vector{Float64}[],
                level_spacings = Vector{Float64}[],
                hist = Histogram[]
                )

for hi = h_range
    @info "Collecting results for h = $(hi)"

    H = XXZ.build_matrix_N(h, L, N; J1=J1, V1=V1, is=is, hs=(hi,), pbc=pbc)
    spec = Hermitian(H) |> eigvals

    @info " Unfolding spectrum"
    cutoff = floor(Int64, length(spec) * 0.1)
    unfolded_spec = XXZ.unfold_spectrum(spec)
    level_spacings = unfolded_spec[2:end] .- unfolded_spec[1:end-1]
    level_spacings = level_spacings[cutoff:end-cutoff]
    hist = fit(Histogram, level_spacings, bins)

    push!(df_h, (hi, unfolded_spec, level_spacings, hist))
end


###
### Animate results
###

time_step = Observable(1)
N = length(h_range)
weights = lift(time_step) do step
    i = to_value(step)
    i = i > N ? 2*N - i + 1 : i
    weights = df_h[i, :hist].weights
    weights ./ sum(weights)
end

label = lift(time_step) do step
    i = to_value(step)
    i = i > N ? 2*N - i + 1 : i
    @sprintf "h = %.2f" df_h[i, :].h
end

title = "Unfolded level statisitics of H = Hxxz + h * σz"
framerate = 30

f = Figure(fontsize=35, resolution=(1400, 1400))
a = Axis(f[1, 1], xlabel="s", ylabel="P(s)", title=title)
centres = (bins[1:end-1] .+ bins[2:end]) ./ 2
barplot!(centres, weights)
text!(label, position=(3, 0.08), textsize=40)
# ylims!(a, 0, 0.1)

record(f, joinpath(plotsdir("XXZ"), "level_statistics_single_impurity.gif"), 1:2*N;
        framerate = framerate) do t
    time_step[] = t
end