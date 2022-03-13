# TODO: Use type parametrisation to generate different hamiltonian functions
# ie use Δ and λ as types and specialise on those.
function apply_H(n::Unsigned, L)
    Δ = 2
    λ = 0.2

    n_aligned = number_of_aligned_neighbours(n, 1, L)
    diag = (-Δ/2) * (2*n_aligned - L)

    if λ > 0
        n_aligned = number_of_aligned_neighbours(n, 2, L)
        diag2 = (-Δ/2) * (2*n_aligned - L)
        diag = diag/(1+λ) + λ * diag2
    end

    output = [(n, diag)]
    for l = 0:L-2
        if ((n & (1 << l)) << 1) ⊻ (n & (1 << (l+1))) != 0 # If n[l] != n[l+1]
            m = flipbits(n, l, l+1)
            push!(output, (m, -1))
        end
    end

    # Periodic boundary condition
    if (n & 1) ⊻ (n >> (L-1)) != 0
        m = flipbits(n, 0, L-1)
        push!(output, (m, -1))
    end

    if λ > 0
        for l = 0:L-3
            if ((n & (1 << l)) << 2) ⊻ (n & (1 << (l+2))) != 0 # If n[l] != n[l+2]
                m = flipbits(n, l, l+2)
                push!(output, (m, -λ))
            end
        end

        # Periodic boundary condition
        if (n & 1) ⊻ ((n >> (L-2)) & 1) != 0
            m = flipbits(n, 0, L-2)
            push!(output, (m, -λ))
        end
        if (n & 2) ⊻ ((n >> (L-2)) & 2) != 0
            m = flipbits(n, 1, L-1)
            push!(output, (m, -λ))
        end
    end

    output
end

function build_basis_N(T::DataType, L::Integer, N::Integer)
    # TODO: lines a and b seem to be allocating while c doesn't 
    # according to julia --track-allocation=user. This could be hitting performance.
    cardinality = binomial(L, N)
    basis = Array{T, 1}(undef, cardinality) # c

    ei = ~((~zero(T)) << N) # a
    for i in 1:cardinality
        basis[i] = ei
        ei = next_basis_element(ei) # b
    end

    basis
end

#======================================================#
# Utility functions
#======================================================#

#=
The below implementation was taken from:
https://math.stackexchange.com/questions/2254151/is-there-a-general-formula-to-generate-all-numbers-with-a-given-binary-hamming

c is all zeros but with a 1 where the the least significant 1 is in e

Adding 1 to the least significant 1 in e causes all neighbouring 1s
to set to 0 and the next 0 to flip to 1. This is r.

(r ⊻ e) contains only the least significant block of 1s in e with the next zero in e
flipped to 1.

((r ⊻ e) >> 2) ÷ c contains all the ones left of c, in e, in the least significant block of 1s
in the least significant bits.

or-ing with r returns the result.

The overall effect is to take the least significant block in e, move its most significant 1 to
the left and moving the rest as far to the right as possible.
=#
"""
Produces the next element in the basis of states with equal number of ones.
"""
function next_basis_element(e::T)::T where T <: Union{UInt32, UInt64}
    c = e & -e
    r = e + c
    (((r ⊻ e) >> 2) ÷ c) | r
end

#ni_equals_nj(n, i, j) = ((n & (1 << i)) << (j-i)) ⊻ (n & (1 << j)) != 0

function flipbits(n::Unsigned, i, j)
    # This function assumes i < j
    m = ((1 << (j - i)) + 1) << i
    n ⊻ m
end

function number_of_aligned_neighbours(n::T, i, L) where T <: Union{UInt64, UInt32}
    N = T === UInt32 ? 32 : 64
    mask = typemax(T) >> (N - L)
    n = n & mask # TODO: can remove this line if we assume n[l] is 0 when l > L
    m = (n << i) + (n >> (L-i))
    aligned = ~(n ⊻ m) & mask
    hamming_weight(aligned)
end

#TODO: Use Holy traits pattern to "parameterise" all 32, 64 bit types.
"""
    hamming_weight(b)

Return the Hamming weight of b. (ie the Hamming distance from all zeros, 
ie the number of 1s in the binary representation of b)
"""
function hamming_weight(b::T) where T <: Union{UInt64, UInt32}
    if T === UInt32       # Binary representation:
        m1  = 0x55555555  # 0101010101010101....
        m2  = 0x33333333  # 0011001100110011....
        m4  = 0x0f0f0f0f  # 0000111100001111....
    else
        m1  = 0x5555555555555555
        m2  = 0x3333333333333333
        m4  = 0x0f0f0f0f0f0f0f0f
    end

    b -= (b >> 1) & m1             # put count of each 2 bits into those 2 bits
    b = (b & m2) + ((b >> 2) & m2) # put count of each 4 bits into those 4 bits 
    b = (b + (b >> 4)) & m4        # put count of each 8 bits into those 8 bits 
    b += b >>  8                   # put count of each 16 bits into their lowest 8 bits
    b += b >> 16                   # put count of each 32 bits into their lowest 8 bits

    if T === UInt64
        b += b >> 32               # put count of each 64 bits into their lowest 8 bits
        return UInt64(127) & b
    else
        return UInt32(63) & b
    end
end

# Below implementation should run faster on CPUs with better multiplcation units.
# function hamming_weight2(b::T) where T <: Union{UInt64, UInt32}
#     if T === UInt32       # Binary representation:
#         h01 = 0x01010101  # 0000000100000001....
#         m1  = 0x55555555  # 0101010101010101....
#         m2  = 0x33333333  # 0011001100110011....
#         m4  = 0x0f0f0f0f  # 0000111100001111....
#     else
#         h01 = 0x0101010101010101
#         m1  = 0x5555555555555555
#         m2  = 0x3333333333333333
#         m4  = 0x0f0f0f0f0f0f0f0f
#     end

#     b -= (b >> 1) & m1             # put count of each 2 bits into those 2 bits
#     b = (b & m2) + ((b >> 2) & m2) # put count of each 4 bits into those 4 bits 
#     b = (b + (b >> 4)) & m4        # put count of each 8 bits into those 8 bits

#     d = T === UInt32 ? 24 : 56
#     return (b * h01) >> d
# end