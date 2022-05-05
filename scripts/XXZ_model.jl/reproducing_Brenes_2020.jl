#=
This script reproduces some of the results presented in Brenes_2020.
=#

using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

#===============================================================#
# Collect operator elements for analysis
#===============================================================#

J1 = 2.0; V1 = 0.55
pbc = false

df = DataFrame(L = Int[], 
               h = Float64[], 
               Es = Vector[], 
               eps = Vector[], 
               Ks = Vector[], 
               Ts = Vector[], 
               Tnm = Matrix[], 
               Jnm=Matrix[]
               )

for L in [10, 12, 14], hi in 0.0:1.0
    @info "Running for L = $(L), hi = $(hi)"
    N = L÷2; hs = (0.1, hi,); is = (0, N,)
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, hs=hs, is=is, pbc)
    F = Hermitian(H) |> eigen
    d = length(F.values)

    i = L÷4; j = i+1
    K = XXZ.build_matrix_N(XXZ.apply_K, L, N, 2.0, i, j)
    T = XXZ.build_matrix_N(XXZ.apply_T, L, N, 1, pbc)
    J = XXZ.build_matrix_N(XXZ.apply_J, L, N, 1, pbc)

    @info "    Computing operator elements"
    K_avrg = F.vectors' * K * F.vectors |> diag
    T_nm = F.vectors' * T * F.vectors
    T_avrg = diag(T_nm)
    J_nm = F.vectors' * J * F.vectors

    En = F.values
    eps = (En .- minimum(En)) ./ (maximum(En) .- minimum(En))
    push!(df, (L, hi, F.values, eps, K_avrg, T_avrg, T_nm, J_nm))
end

#===============================================================#
# Reproduce Figure 1 from Brenes 2020
#===============================================================#

# Compute micro-canonical enemble averages for the local and global kinetic energies.
r = df[df.L .== 14 .&& df.h .== 0, :]
δϵ = 0.02
domain = [(δϵ/2):(δϵ/2):(1 - δϵ/2)...]
mce_avg_K, domain_K = XXZ.coarse_grained_average(r.Ks[1], r.eps[1], domain, δϵ)
mce_avg_T, domain_T = XXZ.coarse_grained_average(r.Ts[1], r.eps[1], domain, δϵ)


f1 = Figure(resolution=(2800, 2000), fontsize=35)

# Plot panel (1, 1)
a = Axis(f1[1, 1], ylabel="Expectation K", title="h = 0")
for d in eachrow(df[df.h .== 0.0, :])
    scatter!(f1[1, 1], d.eps, d.Ks |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[1, 1], domain_K, mce_avg_K |> real, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5


# Plot panel (1, 2)
a = Axis(f1[1, 2], title="h = 1")
for d in eachrow(df[df.h .== 1.0, :])
    scatter!(f1[1, 2], d.eps, d.Ks |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[1, 2], domain_K, mce_avg_K |> real, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5


# Plot panel (2, 1)
a = Axis(f1[2, 1], xlabel="Energy density", ylabel="Expectation T")
for d in eachrow(df[df.h .== 0.0, :])
    scatter!(f1[2, 1], d.eps, d.Ts |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[2, 1], domain_T, mce_avg_T |> real, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5


# Plot panel (2, 2)
a = Axis(f1[2, 2], xlabel="Energy density")
for d in eachrow(df[df.h .== 1.0, :])
    scatter!(f1[2, 2], d.eps, d.Ts |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[2, 2], domain_T, mce_avg_T |> real, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5
axislegend(a, position = :rb)

# save(joinpath(plotsdir("XXZ"), "fig1.png"), f1)


#===============================================================#
# Reproduce Figure 2 from Brenes 2020
#===============================================================#

"""
For the relevant off-diagonal elements of the given operator Oₙₘ,
computes |Oₙₘ|² * N * D. 

The relevant elements are those whose corresponding eigenvectors have
average energy close to zero. 
"""
function onm_squared_ND(Es, Onm)
    ϵ = 0.025 * (maximum(Es) - minimum(Es)) / 2

    ω = Float64[]
    O2ND = Float64[]
    D = length(Es)
    for i = 1:D-1, j = i+1:D
        Ebar = (Es[i] + Es[j])/2
        if abs(Ebar) <= ϵ
            o2nd = abs(Onm[i, j])^2 * D * data.L
            if o2nd != 0.0
                push!(ω, Es[j] - Es[i])
                push!(O2ND, o2nd)
            end
        end
    end

    ω, O2ND
end

f2 = Figure(resolution=(2800, 2000), fontsize=35)


# Plot panel (1, 1)
data = df[df.L .== 14 .&& df.h .== 0, :][1, :]
ω, T2ND = onm_squared_ND(data.Es, data.Tnm)
a = Axis(f2[1, 1], ylabel="|Tnm|^2 ND", title="h = 0", yscale=log10)
scatter!(f2[1, 1], ω, T2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)
avg_T2ND, avg_ω = XXZ.coarse_grained_average(T2ND, ω, [0:25...], 1.0)
lines!(a, avg_ω, avg_T2ND |> real, color=:black, linewidth=5)


# Plot panel (1, 2)
data = df[df.L .== 14 .&& df.h .== 1.0, :][1, :]
ω, T2ND = onm_squared_ND(data.Es, data.Tnm)
a = Axis(f2[1, 2], title="h = 1", yscale=log10)
scatter!(f2[1, 2], ω, T2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)
lines!(a, avg_ω, avg_T2ND |> real, color=:black, linewidth=5)


# Plot panel (2, 1)
data = df[df.L .== 14 .&& df.h .== 0, :][1, :]
ω, J2ND = onm_squared_ND(data.Es, data.Jnm)
a = Axis(f2[2, 1], ylabel="|Jnm|^2 ND", yscale=log10, xlabel="ω")
scatter!(f2[2, 1], ω, J2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)
avg_J2ND, avg_ω = XXZ.coarse_grained_average(J2ND, ω, [0:25...], 1.0)
lines!(a, avg_ω, avg_J2ND, color=:black, linewidth=5)


# Plot panel (2, 2)
data = df[df.L .== 14 .&& df.h .== 1.0, :][1, :]
ω, J2ND = onm_squared_ND(data.Es, data.Jnm)
a = Axis(f2[2, 2], yscale=log10, xlabel="ω")
scatter!(f2[2, 2], ω, J2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)
lines!(a, avg_ω, avg_J2ND, color=:black, linewidth=5)

# save(joinpath(plotsdir("XXZ"), "fig2.png"), f2)