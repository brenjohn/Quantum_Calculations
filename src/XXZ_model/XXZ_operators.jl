#=
This file defines operators for the XXZ model.

Here, an operator is defined by a function which returns the image of a given basis state.
The image of the basis state is stored in an `output` vector holding state-weight tuples.
=#

#===========================================================================#
# The XXZ Hamiltonian operator
#===========================================================================#

# TODO: preallocate output vector and reuse it: apply_H!(output, n, L)
# TODO: use singleton types and multiple dispatch to remove the "if J !=0" lines in hopping and interaction terms.

"""
Returns the image of the state `n` under the action of the XXZ Hamiltonian.

 H = J1/2 * ∑ₙₙ (σˣᵢσˣⱼ + σʸᵢσʸⱼ) + V1 * ∑ₙₙ σᶻᵢσᶻⱼ
   + J2/2 * ∑ₙₙₙ(σˣᵢσˣⱼ + σʸᵢσʸⱼ) + V2 * ∑ₙₙₙσᶻᵢσᶻⱼ
   + ∑ₛhₛσᶻₛ

   = J1 * ∑ₙₙ (bᵢb⁺ⱼ + b⁺ᵢbⱼ) + V1 * ∑ₙₙ (2nᵢ - 1)(2nⱼ - 1)
   + J2 * ∑ₙₙₙ(bᵢb⁺ⱼ + b⁺ᵢbⱼ) + V2 * ∑ₙₙₙ(2nᵢ - 1)(2nⱼ - 1)
   + ∑ₛhₛ(2nₛ - 1)

# arguments
- `n::Unsigned`: An Unsigned integer representing a computational basis state.
- `L`: The number of sites in the XXZ chain to be considered (ie the number of bits in `n` to account for.)

# keyword arguments
- `J1`: The strength of nearest neighbour hopping.
- `V1`: The strength of nearest neighbour interaction.
- `J2`: The strength of next nearest neighbour hopping.
- `V2`: The strength of next nearest neighbour interaction.
- `hs`: An iterable containing the single site impurity strengths.
- `is`: An iterable containing the locations of single site impurities.
- `pbs`: `true` if periodic boundary conditions are to be imposed.
"""
function apply_H(n::T, L;
                J1 = 1.0, 
                V1 = 1.0, 
                J2 = 0.0, 
                V2 = 0.0,
                hs = (),
                is = (), 
                pbc= true
                ) where T <: Unsigned

    output = Tuple{T, Float64}[]

    # Apply nearest and next-nearest neighbour hopping.
    J1 != 0 && hopping_term!(output, J1, 1, n, L, pbc)
    J2 != 0 && hopping_term!(output, J2, 2, n, L, pbc)

    # Apply nearest and next-nearest neighbour interaction terms.
    V1 != 0 && neighbour_interaction_term!(output, V1, 1, n, L, pbc)
    V2 != 0 && neighbour_interaction_term!(output, V2, 2, n, L, pbc)

    # Apply sigma-z impurities
    for (h, i) in zip(hs, is)
        h != 0 && single_site_impurity!(output, h, i, n)
    end

    output
end

"""
Uses the parameterisation of the XXZ Hamiltonian used in the Jung-Hoon 2020 tutorial.
"""
function apply_H(n::Unsigned, L, Δ, λ)
    J1 = -1/(1+λ)
    J2 = -λ/(1+λ)
    V1 = -0.5*Δ/(1+λ)
    V2 = -0.5*λ*Δ/(1+λ)
    apply_H(n, L; J1=J1, V1=V1, J2=J2, V2=V2)
end

"""
Generate the action of `d`-nearest neighbour hopping, with strength `J`, on 
the state `n` of a system of length `L` and push it to the `output` vector.
"""
function hopping_term!(output, J, d, n, L, pbc)
    for l = 0:L-1-d
        apply_K!(output, n, J, l, l+d)
    end

    # Periodic boundary condition
    if pbc
        for b = 0:d-1
            apply_K!(output, n, J, b, L-d+b)
        end
    end
    output
end

"""
Generate the action of `d`-nearest neighbour interaction, with strength `V`, on 
the state `n` of a system of length `L` and push it to the `output` vector.
"""
function neighbour_interaction_term!(output, V, d, n, L, pbc)
    n_aligned = number_of_aligned_neighbours(n, d, L, pbc)
    n_neighbours = pbc ? L : L - d
    diag = V * (2*n_aligned - n_neighbours)
    push!(output, (n, diag))
    output
end

apply_site_impurity(n::T, L, h, i) where T <: Unsigned = single_site_impurity!(Tuple{T, Float64}[], h, i, n)

function single_site_impurity!(output, h, i, n)
    ni = (n >> i) & 1
    s = 2 * ni - 1
    push!(output, (n, s * h))
end

apply_sigma_zz(n::T, L, h, i, j) where T <: Unsigned = sigma_zz!(Tuple{T, Float64}[], h, i, j, n)

function sigma_zz!(output, h, i, j, n)
    ni = (n >> i) & 1
    nj = (n >> j) & 1
    s = 2*(ni == nj) - 1
    push!(output, (n, s * h))
end


#===========================================================================#
# Here, we define the operators A and B as defined in the Jung-Hoon_2020 
# tutorial. These are:
#     - A = Zero momentum distribution 
#     - B = Nearest neigbour interaction energy density
#===========================================================================#

"""
Returns the image of n under the zero momentum distribution operator.

    A = 1/L ∑ᵢⱼ(σ⁺ᵢσ⁻ⱼ) = 1/L ∑ᵢⱼ(b⁺ᵢbⱼ)
"""
function apply_A(n::Unsigned, L)
    weight = 1 / L
    diag = hamming_weight(n) / L
    output = [(n, diag)]

    for l in 0:L-2
        for m in l+1:L-1
            if bits_differ(n, l, m)
                o = flipbits(n, l, m)
                push!(output, (o, weight))
            end
        end
    end
    output
end

"""
Returns the image of n under the nearest neigbour interaction energy density operator.

    B = 1/4L ∑ₙₙ(σᶻᵢ+1)(σᶻⱼ+1) = 1/L ∑ₙₙnᵢnⱼ
"""
function apply_B(n::Unsigned, L)
    nn = translate(n, L)
    nn = n & nn
    weight = hamming_weight(nn) / L
    [(n, weight)]
end

#===========================================================================#
# Below we define the operators discussed in Brenes_2020:
#     - K = local kinetic energy
#     - T = average kinetic energy
#     - J = spin current
#===========================================================================#

"""
Returns the image of the basis state `n` under the local kinetic energy operator:

    K = J/2 * (σˣᵢσˣⱼ + σʸᵢσʸⱼ) = J * (bᵢb⁺ⱼ + b⁺ᵢbⱼ)
"""
apply_K(n::T, L, args...) where T <: Unsigned = apply_K!(Tuple{T, Float64}[], n, args...)

function apply_K!(output, n::T, J, i, j) where T <: Unsigned
    if bits_differ(n, i, j)
        m = flipbits(n, i, j)
        push!(output, (m, J))
    end
    output
end

"""
Returns the image of the basis state `n` under the `d`-nn average kinetic energy operator:

    T = ∑ₙₙ(σˣᵢσˣⱼ + σʸᵢσʸⱼ) / L = 2/L * ∑ₙₙ(bᵢb⁺ⱼ + b⁺ᵢbⱼ)
"""
apply_T(n::T, L, d, pbc=false) where T <: Unsigned = hopping_term!(Tuple{T, Float64}[], 2/L, d, n, L, pbc)

"""
Returns the image of the basis state `n` under the spin current operator:

    J = ∑ₙₙ(σˣᵢσʸⱼ - σʸᵢσˣⱼ) / L = 2i/L * ∑ₙₙ(bᵢb⁺ⱼ - b⁺ᵢbⱼ)
"""
function apply_J(n::T, L, d=1, pbc=false) where T <: Unsigned
    output = Tuple{T, ComplexF64}[]
    Jp = 2im/L; Jm = -2im/L

    for l = 0:L-1-d
        if bits_differ(n, l, l+d)
            m = flipbits(n, l, l+d)
            J = m > n ? Jp : Jm # m > n => a 1 has moved to the right. 
            push!(output, (m, J))
        end
    end

    # Periodic boundary conditions
    if pbc
        for b = 0:d-1
            if bits_differ(n, b, L-d+b)
                m = flipbits(n, b, L-d+b)
                J = m < n ? Jp : Jm # m < n => a 1 has moved to the right across the boundary.
                push!(output, (m, J))
            end
        end
    end

    output
end

#===========================================================================#
# Below
#===========================================================================#

spacial_correlation(n, L, i, j) = [(n, -2 * bits_differ(n, i, j) + 1)]