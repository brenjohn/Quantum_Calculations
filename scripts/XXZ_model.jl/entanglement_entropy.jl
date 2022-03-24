using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0
L = 20

basis = XXZ.build_MSS_basis(L)
# index_map = Dict(e => i for (i, (e, p)) in enumerate(basis))

H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)
