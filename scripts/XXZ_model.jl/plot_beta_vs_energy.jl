using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie
using Distributed

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0

df = DataFrame(L = Int[], betas = Vector[], Es = Vector[])
β_range = 0.0:0.05:2.5
L_range = [8, 12, 16, 20]

addprocs(5)
for L in L_range
    @info "Running for L = $L"
    H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)
    E = Hermitian(H) |> eigvals
    E_β = zeros(length(β_range))

    for (j, β) in enumerate(β_range)
        f = e -> exp(-β*e)
        exp_βE = pmap(f, E)

        numerator = @distributed (+) for i = 1:length(E)
            E[i] * exp_βE[i]
        end
        Z = sum(exp_βE)
        E_β[j] = numerator/Z
    end

    push!(df, (L, β_range, E_β))
end
rmprocs(workers())

###
### Plot results
###

f = Figure(fontsize=35)
a = Axis(f[1, 1], xlabel="E/L", ylabel="β", title="Energy density vs inverse temperature.")
for d in eachrow(df)
    lines!(f[1, 1], d.Es/d.L, d.betas, linewidth = 3, label="L = $(d.L)")
end
a.yticks = 0:2
axislegend(a, position = :rt)

save(joinpath(plotsdir("XXZ"), "beta_vs_energy.png"), f)