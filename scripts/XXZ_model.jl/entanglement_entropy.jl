using DrWatson
using LinearAlgebra
using DataFrames
using GLMakie, Makie

include(srcdir("XXZ_model/XXZ_model.jl"))

Δ = 0.5
λ = 1.0

df = DataFrame(L = Int[], L1s = Vector[], entropies = Vector[])


for L in [8, 12, 16, 20]
    @info "Running for L = $L"
    # Prepare the state to be examined.
    basis = XXZ.build_MSS_basis(L)
    H = XXZ.build_matrix_MSS(XXZ.apply_H, L, Δ, λ)
    F = Hermitian(H) |> eigen
    α = map(abs, F.values) |> argmin
    V = F.vectors[:, α]

    # Compute the entanglement entropy for different partition sizes.
    state = XXZ.MSS_to_full_state(ψ0, L, basis)
    entropies = zeros(L+1)
    for L1 = 0:L
        entropies[L1+1] = XXZ.entanglement_entropy(state, L, L1)
    end

    push!(df, (L, 0:L, entropies))
end

###
### Plot the results
###

f = Figure(fontsize=35)
a = Axis(f[1, 1], xlabel="L1", ylabel="Sₑ", title="Entanglement Entropy")
for d in eachrow(df)
    scatter!(d.L1s, d.entropies, label="L = $(d.L)")
end
axislegend(a, position = :rt)