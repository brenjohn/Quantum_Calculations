using DrWatson
using LinearAlgebra
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 0.0
L = 12

###
### Create Hamiltonians
###

h    = XXZ.apply_H
H    = XXZ.build_matrix(h, L, Δ, λ)
HN   = XXZ.build_matrix_N(h, L, L÷2, Δ, λ)
HNk  = XXZ.build_matrix_Nk(h, L, L÷2, 0, Δ, λ)
HMSS = XXZ.build_matrix_MSS(h, L, Δ, λ)


###
### Find the energy spectra
###

spec    = Hermitian(H)    |> eigvals
specN   = Hermitian(HN)   |> eigvals
specNk  = Hermitian(HNk)  |> eigvals
specMSS = Hermitian(HMSS) |> eigvals

###
### Plot spectra
###

f = Figure(fontsize=21)
a = Axis(f[1, 1], xlabel="α/|S|", ylabel="α", title="Spectra for H, HN, HNk, HMSS with L=$L Δ=$Δ λ=$λ")

for (i, s, l) in ((1, spec, "full"), (2, specN, "SN"), (3, specNk, "SNk"), (4, specMSS, "SMSS"))
    scatter!(LinRange(0, 1, length(s)), s, label=l, markersize = 20 - 4*i)
end

a.yticks = -4:6
axislegend(a, position = :lt)

# save(joinpath(plotsdir("XXZ"), "Symmetry_sector_spectra.png"), f)

# f = Figure(resolution=(1600, 1600))
# a = Axis(f[1, 1], title="L=$L Δ=$Δ λ=$λ")
# spy!(sparse(H[:, end:-1:1]), markersize=5)
# hidedecorations!(a)
# save(joinpath(plotsdir("XXZ"), "hamiltonian_full_delta_0-5_lambda_0.png"), f)