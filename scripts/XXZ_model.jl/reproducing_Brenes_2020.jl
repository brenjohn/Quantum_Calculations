using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

J1 = 2.0; V1 = 0.55
J2 = 0.0; V2 = 0.0
pbc = false

#===============================================================#
# Collect operator elements for analysis
#===============================================================#

df = DataFrame(L = Int[], h = Float64[], Es = Vector[], eps = Vector[], Ks = Vector[], Ts = Vector[], Tnm = Matrix[], Jnm=Matrix[])

for L in [14], hi in 0.0:1.0
    @info "Running for L = $(L), hi = $(hi)"
    N = L÷2; hs = (0.1, hi,); is = (0, N,)
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)
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
# Compute ensemble averages
#===============================================================#

function coarse_grained_average(xs, ys, domain, dx)
    average = zeros(Float64, length(domain))

    for (i, xi) in enumerate(domain)
        f = x -> (xi - dx/2) < x < (xi + dx/2)
        inds = findall(f, xs)
        average[i] = sum(ys[inds]) / length(inds) |> real
    end

    inds = findall(!isnan, average)
    average[inds], domain[inds]
end

function microcanonical_ensemble_average(eps, values, δϵ)
    domain = [(δϵ/2):(δϵ/2):(1 - δϵ/2)...]
    mce_average = zeros(Float64, length(domain))

    for (i, ϵ) in enumerate(domain)
        f = e -> (ϵ - δϵ/2) < e < (ϵ + δϵ/2)
        inds = findall(f, eps)
        mce_average[i] = sum(values[inds]) / length(inds) |> real
    end

    mce_average, domain
end

δϵ = 0.02
r = df[df.L .== 14 .&& df.h .== 0, :]
mce_avg_K, domain_K = microcanonical_ensemble_average(r.eps[1], r.Ks[1], δϵ)
inds = findall(!isnan, mce_avg_K)
domain_K = domain_K[inds]
mce_avg_K = mce_avg_K[inds]

δϵ = 0.02
r = df[df.L .== 14 .&& df.h .== 0, :]
mce_avg_T, domain_T = microcanonical_ensemble_average(r.eps[1], r.Ts[1], δϵ)
inds = findall(!isnan, mce_avg_T)
domain_T = domain_T[inds]
mce_avg_T = mce_avg_T[inds]


#===============================================================#
# Plot the results
#===============================================================#

f1 = Figure(resolution=(2800, 2000), fontsize=35)

a = Axis(f1[1, 1], ylabel="Expectation K", title="h = 0")
for d in eachrow(df[df.h .== 0.0, :])
    scatter!(f1[1, 1], d.eps, d.Ks |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[1, 1], domain_K, mce_avg_K, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5

a = Axis(f1[1, 2], title="h = 1")
for d in eachrow(df[df.h .== 1.0, :])
    scatter!(f1[1, 2], d.eps, d.Ks |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[1, 2], domain_K, mce_avg_K, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5

#====#

a = Axis(f1[2, 1], xlabel="Energy density", ylabel="Expectation T")
for d in eachrow(df[df.h .== 0.0, :])
    scatter!(f1[2, 1], d.eps, d.Ts |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[2, 1], domain_T, mce_avg_T, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5

a = Axis(f1[2, 2], xlabel="Energy density")
for d in eachrow(df[df.h .== 1.0, :])
    scatter!(f1[2, 2], d.eps, d.Ts |> real, markersize = 28 - d.L, label="L = $(d.L)")
end
lines!(f1[2, 2], domain_T, mce_avg_T, color=:black, linewidth=5)
a.yticks = -1.5:0.5:1.5
axislegend(a, position = :rb)

# save(joinpath(plotsdir("XXZ"), "eigenstate_expectation_vs_L.png"), f)


#===============================================================#
# Plot the results
#===============================================================#

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

a = Axis(f2[1, 1], ylabel="|Tnm|^2 ND", title="h = 0", yscale=log10)
data = df[1, :]
ω, T2ND = onm_squared_ND(data.Es, data.Tnm)
scatter!(f2[1, 1], ω, T2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)

avg_T2ND, avg_ω = coarse_grained_average(ω, T2ND, [0:25...], 1.0)
lines!(a, avg_ω, avg_T2ND, color=:black, linewidth=5)

a = Axis(f2[1, 2], title="h = 1", yscale=log10)
data = df[2, :]
ω, T2ND = onm_squared_ND(data.Es, data.Tnm)
scatter!(f2[1, 2], ω, T2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)

lines!(a, avg_ω, avg_T2ND, color=:black, linewidth=5)

save(joinpath(plotsdir("XXZ"), "fig1.png"), f1)

#=============#

a = Axis(f2[2, 1], ylabel="|Jnm|^2 ND", yscale=log10, xlabel="ω")
data = df[1, :]
ω, J2ND = onm_squared_ND(data.Es, data.Jnm)
scatter!(f2[2, 1], ω, J2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)

avg_J2ND, avg_ω = coarse_grained_average(ω, J2ND, [0:25...], 1.0)
lines!(a, avg_ω, avg_J2ND, color=:black, linewidth=5)

a = Axis(f2[2, 2], yscale=log10, xlabel="ω")
data = df[2, :]
ω, J2ND = onm_squared_ND(data.Es, data.Jnm)
scatter!(f2[2, 2], ω, J2ND, markersize = 20 - data.L, label="L = $(data.L)")
ylims!(a, 10^(-25), 10^3)

lines!(a, avg_ω, avg_J2ND, color=:black, linewidth=5)

save(joinpath(plotsdir("XXZ"), "fig2.png"), f2)