using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0

df = DataFrame(L = Int[], Es = Vector[], As = Vector[], Bs = Vector[])

for L in [8, 12, 16, 20]
    @info "Running for L = $L"
    H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)
    F = Hermitian(H) |> eigen
    d = length(F.values)

    A = XXZ.build_matrix_MSS(XXZ.apply_A, L)
    B = XXZ.build_matrix_MSS(XXZ.apply_B, L)

    A_avrg = zeros(d)
    B_avrg = zeros(d)

    for (i, ev) in enumerate(eachcol(F.vectors))
        A_avrg[i] = ev' * A * ev
        B_avrg[i] = ev' * B * ev
    end

    push!(df, (L, F.values, A_avrg, B_avrg))
end

###
### Plot spectra
###

f = Figure(resolution=(2800, 1200), fontsize=35)

a = Axis(f[1, 1], xlabel="E/L", ylabel="Expectation", title="Energy eigenstate expectation of zero momentum distribution")
for d in eachrow(df)
    scatter!(f[1, 1], d.Es/d.L, d.As, markersize = 28 - d.L, label="L = $(d.L)")
end
a.yticks = 0:6
axislegend(a, position = :rt)



a = Axis(f[1, 2], xlabel="E/L", ylabel="Expectation", title="Energy eigenstate expectation of nn-interaction energy density")
for d in eachrow(df)
    scatter!(f[1, 2], d.Es/d.L, d.Bs, markersize = 28 - d.L, label="L = $(d.L)")
end
a.yticks = 0.12:0.03:0.27
axislegend(a, position = :lb)