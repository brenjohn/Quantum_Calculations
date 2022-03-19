using DrWatson
using SparseArrays
using LinearAlgebra

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 2.0
λ = 0.0
L = 8

H = XXZ.build_H(L, Δ, λ)

HN = XXZ.build_HN(L, L÷2, Δ, λ)

HNk = XXZ.build_HNk(L, L÷2, 0, Δ, λ)

HMSS = XXZ.build_HMSS(L, Δ, λ)