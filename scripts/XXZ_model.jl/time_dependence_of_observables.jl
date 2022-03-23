using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0

dt = 0.5
t_range = dt:dt:50
L_range = [12, 16, 20]

df = DataFrame(L = Int[], ts = Vector[], As = Vector[], Bs = Vector[])


for L in L_range
    @info "Running for L = $L"
    basis = XXZ.build_MSS_basis(L)
    index_map = Dict(e => i for (i, (e, p)) in enumerate(basis))

    ψ0 = zeros(length(basis))
    ψ0[end] = 1

    A = XXZ.build_matrix_MSS(XXZ.apply_A, L)
    B = XXZ.build_matrix_MSS(XXZ.apply_B, L)
    H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)

    F = Hermitian(H) |> eigen
    C0 = F.vectors' * ψ0
    Ct = zeros(ComplexF64, length(C0), length(t_range) + 1)
    Ct[:, 1] = C0

    U = cis.(-dt * F.values) |> Diagonal
    for (i, t) in enumerate(t_range)
        Ct[:, i+1] = U * Ct[:, i]
    end

    ψt = F.vectors * Ct

    A_avrg = zeros(length(t_range) + 1)
    B_avrg = zeros(length(t_range) + 1)

    for (i, ψ) in enumerate(eachcol(ψt))
        A_avrg[i] = ψ' * A * ψ |> real
        B_avrg[i] = ψ' * B * ψ |> real
    end

    push!(df, (L, [0, t_range...], A_avrg, B_avrg))
end


###
### Plot results
###

f = Figure()
a = Axis(f[1, 1], xlabel="t", ylabel="Expectation", title="Evolution of expectation values in time.")
for (i, d) in enumerate(eachrow(df))
    lines!(d.ts, d.As, label="A: L = $(d.L)", linewidth = 2i)
    lines!(d.ts, d.Bs, label="B: L = $(d.L)", linewidth = 2i)
end
axislegend(a, position = :rb)