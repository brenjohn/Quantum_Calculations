using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0
L = 12
ϵ = 0.3
N_steps = 50

ψ0 = zeros(ComplexF64, 2^L)
n = ~(typemax(UInt32) << L) & 0x55555555
m = XXZ.translate(n, L)
ψ0[n + 1] = 1/sqrt(2)
ψ0[m + 1] = 1/sqrt(2)

A = XXZ.build_matrix(XXZ.apply_A, L)
B = XXZ.build_matrix(XXZ.apply_B, L)

###
### Exact time evolution.
###

H = XXZ.build_matrix(XXZ.apply_H, L, Δ, λ)

F = Hermitian(H) |> eigen
C0 = F.vectors' * ψ0
Ct = zeros(ComplexF64, length(C0), N_steps)
Ct[:, 1] = C0

U = cis.(-ϵ * F.values) |> Diagonal
for i in 1:N_steps-1
    Ct[:, i+1] = U * Ct[:, i]
end

ψt = F.vectors * Ct
Ext_As = [ψ' * A * ψ |> real for ψ in eachcol(ψt)]
Ext_Bs = [ψ' * B * ψ |> real for ψ in eachcol(ψt)]

###
### Lie-Trotter-Suzuki evolution.
###

ψt = XXZ.LTS_evolution(ψ0, L, Δ, λ, ϵ, N_steps)
LTS_As = [ψ' * A * ψ |> real for ψ in eachcol(ψt)]
LTS_Bs = [ψ' * B * ψ |> real for ψ in eachcol(ψt)]

###
### Plot comparison.
###

f = Figure()
a = Axis(f[1, 1], xlabel="Time", ylabel="Expectation", title="Time evolution of expectation values.")
lines!(0:ϵ:ϵ*(N_steps-1), LTS_As, label="LTS A")
lines!(0:ϵ:ϵ*(N_steps-1), LTS_Bs, label="LTS B")

lines!(0:ϵ:ϵ*(N_steps-1), Ext_As, label="Exact A")
lines!(0:ϵ:ϵ*(N_steps-1), Ext_Bs, label="Exact B")
axislegend(a, position = :rb)

save(joinpath(plotsdir("XXZ"), "LTS_evolution_comparison.png"), f)