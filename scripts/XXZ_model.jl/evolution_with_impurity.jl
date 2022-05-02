using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

function time_evolve(Os, F, t_range, ψᵢ)
    Ct = zeros(ComplexF64, length(ψᵢ), length(t_range) + 1)
    Ct[:, 1] = F.vectors' * ψᵢ

    U = cis.(-dt * F.values) |> Diagonal
    for i in 1:length(t_range)
        Ct[:, i+1] = U * Ct[:, i]
    end
    ψt = F.vectors * Ct #TODO: take expectation values in eigenbasis

    O_avrg = zeros(ComplexF64, length(Os), length(t_range) + 1)
    for (j, O) in enumerate(Os)
        for (i, ψ) in enumerate(eachcol(ψt))
            O_avrg[j, i] = ψ' * O * ψ
        end
    end
    O_avrg
end

#==============#

J1 = 2.0; V1 = 0.55
pbc = false
L = 14; N = L ÷ 2
hs = (1.0,); is = (L÷2,)

H_xxz = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, pbc=pbc)
F_xxz = Hermitian(H_xxz) |> eigen

H_si = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, hs=hs, is=is, pbc=pbc)
F_si = Hermitian(H_si) |> eigen

basis = XXZ.build_basis_N(UInt32, L, N)
basis_index = Dict(basis .=> 1:length(basis))
neel_state = 0x55555555 >> (32 - L)
ψᵢ = zeros(ComplexF64, length(basis))
ψᵢ[basis_index[neel_state]] = 1.0 + 0.0im

i = L ÷ 4
K = XXZ.build_matrix_N(XXZ.apply_K, L, N, 1.0, i, i+1)
T = XXZ.build_matrix_N(XXZ.apply_T, L, N, 1, pbc)
SI = XXZ.build_matrix_N(XXZ.appy_site_impurity, L, N, 1.0, i)
J = XXZ.build_matrix_N(XXZ.apply_J, L, N, 1, pbc)


dt = 0.5
t_range = dt:dt:200

Ot_xxz = time_evolve([K, T, SI, J], F_xxz, t_range, ψᵢ)
Ot_si = time_evolve([K, T, SI, J], F_si, t_range, ψᵢ)

#=========#

f = Figure(resolution=(2000, 1400), fontsize=35)

a = Axis(f[1, 1], xlabel="t", ylabel="Expectation", title="Local Kinetic Energy")
lines!(a, [0.0; t_range], Ot_xxz[1, :] |> real, label="XXZ", linewidth=3)
lines!(a, [0.0; t_range], Ot_si[1, :] |> real, label="SI", linewidth=3)

a = Axis(f[1, 2], xlabel="t", ylabel="Expectation", title="Average Kinetic Energy")
lines!(a, [0.0; t_range], Ot_xxz[2, :] |> real, label="XXZ", linewidth=3)
lines!(a, [0.0; t_range], Ot_si[2, :] |> real, label="SI", linewidth=3)

a = Axis(f[2, 1], xlabel="t", ylabel="Expectation", title="Single Site Magnetisation σᶻᵢ")
lines!(a, [0.0; t_range], Ot_xxz[3, :] |> real, label="XXZ", linewidth=3)
lines!(a, [0.0; t_range], Ot_si[3, :] |> real, label="SI", linewidth=3)

a = Axis(f[2, 2], xlabel="t", ylabel="Expectation", title="Spin Current Operator")
lines!(a, [0.0; t_range], Ot_xxz[4, :] |> real, label="XXZ", linewidth=3)
lines!(a, [0.0; t_range], Ot_si[4, :] |> real, label="SI", linewidth=3)

axislegend(a, position = :rt)

save(joinpath(plotsdir("XXZ"), "evolution_of_observables_with_single_site_impurity.png"), f)