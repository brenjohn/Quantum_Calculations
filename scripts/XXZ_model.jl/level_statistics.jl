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
L = 20
N = L ÷ 3

h = XXZ.apply_H

function unfold_spectrum(spectrum)
    staircase(E) = sum(En -> En <= E, spectrum)
    domain = range(start = spectrum[1], stop = spectrum[end], length=2*length(spectrum))
    # domain = [spectrum[i] for i in [[1:5:length(spectrum)...]; 273]]
    image = staircase.(domain)

    model(x, params) = sum(i -> x.^(i-1) .* params[i], 1:length(params))
    params = zeros(15)
    best_fit = curve_fit(model, domain, image, params)
    unfolded_spectrum = model(spectrum, coef(best_fit))

    unfolded_spectrum
end

bins_unfolded = 0.0:0.1:4.0
centres_unfolded = (bins_unfolded[1:end-1] .+ bins_unfolded[2:end]) ./ 2

df_λ = DataFrame(
    λ = Float64[],
    df_k = DataFrame[],
    avrg_weigths = Vector{Float64}[]
    )

λ_range = 0.0:0.001:0.2
for λ = λ_range
    df_k = DataFrame(
                k = Int[],
                unfolded_spec = Vector{Float64}[],
                level_spacings = Vector{Float64}[],
                hist = Histogram[]
                )
    @info "Collecting results for λ = $λ"

    for k in 1:(L÷2)-1
        @info " Running for k = $k"
        H = XXZ.build_matrix_Nk(h, L, N, k, Δ, λ)
        spec = Hermitian(H) |> eigvals

        @info " Unfolding spectrum"
        cutoff = floor(Int64, length(spec) * 0.1)
        unfolded_spec = unfold_spectrum(spec)
        level_spacings = unfolded_spec[2:end] .- unfolded_spec[1:end-1]
        level_spacings = level_spacings[cutoff:end-cutoff]
        hist = fit(Histogram, level_spacings, bins_unfolded)

        push!(df_k, (k, unfolded_spec, level_spacings, hist))
    end

    avrg_weigths = sum(r -> r.hist.weights, eachrow(df_k)) ./ L
    avrg_weigths /= sum(avrg_weigths)

    push!(df_λ, (λ, df_k, avrg_weigths))
end


###
### Animate results
###

time_step = Observable(1)
N = length(λ_range)
weights = lift(time_step) do step
    i = to_value(step)
    i = i > N ? 2*N - i + 1 : i
    df_λ[i, :].avrg_weigths
end

label = lift(time_step) do step
    i = to_value(step)
    i = i > N ? 2*N - i + 1 : i
    @sprintf "λ = %.2f" df_λ[i, :].λ
end

title = "Unfolded level statisitics of H = (Hnn + λ Hnnn)/(1+λ)"
framerate = 30

f = Figure(fontsize=35, resolution=(1400, 1400))
a = Axis(f[1, 1], xlabel="s", ylabel="P(s)", title=title)
barplot!(centres_unfolded, weights)
text!(label, position=(3, 0.08), textsize=40)
ylims!(a, 0, 0.1)

record(f, joinpath(plotsdir("XXZ"), "level_statistics_evolution.gif"), 1:2*N;
        framerate = framerate) do t
    time_step[] = t
end