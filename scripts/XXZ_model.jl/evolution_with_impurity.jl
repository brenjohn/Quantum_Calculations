#=
This script time evolves an initial state under both the XXZ Hamiltonian and
the same Hamiltonian with an impurity on a single site. The expectation value
of various observables is recorded during the evolutions and compared in a
plot.
=#

using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

#===============================================================#
# Utility functions for the simulation.
#===============================================================#

function expectation_values(Os, ψt)
    O_avrg = zeros(ComplexF64, length(Os), size(ψt)[2])
    for (j, O) in enumerate(Os)
        for (i, ψ) in enumerate(eachcol(ψt))
            O_avrg[j, i] = ψ' * O * ψ
        end
    end
    O_avrg
end

#===============================================================#
# Perform the simulation and gather results.
#===============================================================#

# Set up system parameters.
J1 = 2.0; V1 = 0.55
pbc = false
L = 14; N = L ÷ 2
hs = (1.0,); is = (L÷2,)

# Compute Hamiltonians with ad without the impurity.
H_xxz = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, pbc=pbc)
F_xxz = Hermitian(H_xxz) |> eigen

H_si = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, hs=hs, is=is, pbc=pbc)
F_si = Hermitian(H_si) |> eigen

# Create the initial state to time evolve.
basis = XXZ.build_basis_N(UInt32, L, N)
basis_index = Dict(basis .=> 1:length(basis))
neel_state = 0x55555555 >> (32 - L)
ψᵢ = zeros(ComplexF64, length(basis))
ψᵢ[basis_index[neel_state]] = 1.0 + 0.0im

# Create operator matrices to compute expectation values for.
i = L ÷ 4
K  = XXZ.build_matrix_N(XXZ.apply_K, L, N, 1.0, i, i+1)
T  = XXZ.build_matrix_N(XXZ.apply_T, L, N, 1, pbc)
SI = XXZ.build_matrix_N(XXZ.apply_site_impurity, L, N, 1.0, i)
J  = XXZ.build_matrix_N(XXZ.apply_J, L, N, 1, pbc)
T2 = XXZ.build_matrix_N(XXZ.apply_T, L, N, 2, pbc)
B  = XXZ.build_matrix_N(XXZ.apply_B, L, N)

# Time evolve the initial state.
dt = 0.5; steps = 200 ÷ dt |> Int
ψₜ_xxz = XXZ.time_evolve(ψᵢ, F_xxz, dt, steps)
ψₜ_si  = XXZ.time_evolve(ψᵢ, F_si, dt, steps)

# Compute expectation values.
obs = [K, T, SI, J, T2, B]
Ot_xxz = expectation_values(obs, ψₜ_xxz)
Ot_si  = expectation_values(obs, ψₜ_si)
diag_averages_xxz = XXZ.diagonal_ensemble_average(obs, ψᵢ, F_xxz)
diag_averages_si  = XXZ.diagonal_ensemble_average(obs, ψᵢ, F_si)
micro_averages    = XXZ.micro_canonical_ensemble_average(obs, ψᵢ, F_xxz)

#===============================================================#
# Plot results.
#===============================================================#

struct PlotData
    data_xxz
    data_si
    diag_average_xxz
    diag_average_si
    micro_average
    title
end

function plot_data(Ot_xxz, Ot_si, diag_xxz, diag_si, micro, titles)
    [PlotData(x...) for x in zip(eachrow.([Ot_xxz, Ot_si])..., diag_xxz, diag_si, micro, titles)]
end

titles = ["Local Kinetic Energy", "Average Kinetic Energy", "Single Site Magnetisation σᶻᵢ", 
          "Spin Current Operator", "NNN average kinetic energy", "NN interaction energy"]
my_plot_data = plot_data(Ot_xxz, Ot_si, diag_averages_xxz, diag_averages_si, micro_averages, titles)

f = Figure(resolution=(2000, 1400), fontsize=35)
t_range = 0.0:dt:dt*steps
for (pane, pd) in zip(CartesianIndices((2, 3)), my_plot_data)
    pane = Tuple(pane)
    a = Axis(f[pane...], xlabel="t", ylabel="Expectation", title=pd.title)
    lines!(a, t_range, pd.data_xxz |> real, label="XXZ", linewidth=3)
    lines!(a, t_range, pd.data_si |> real, label="SI", linewidth=3)
    lines!(a, [0.0, dt*steps], real(pd.diag_average_xxz) * [1, 1], label="Diag xxz", linewidth=7, linestyle=:dash, color=:black)
    lines!(a, [0.0, dt*steps], real(pd.diag_average_si) * [1, 1], label="Diag si", linewidth=7, linestyle=:dot, color=:black)
    lines!(a, [0.0, dt*steps], real(pd.micro_average) * [1, 1], label="MCE", linewidth=7, color=:black)
end

# save(joinpath(plotsdir("XXZ"), "evolution_of_observables_with_single_site_impurity.png"), f)