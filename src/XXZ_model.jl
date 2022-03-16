# TODO: Use type parametrisation to generate different hamiltonian functions
# ie use Δ and λ as types and specialise on those.

# TODO: preallocate output vector and reuse it: apply_H!(output, n, L)
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

#======================================================#
# Building HN
#======================================================#

function build_basis_N(T::K, L::Integer, N::Integer) where K <: DataType
    cardinality = binomial(L, N)
    basis = zeros(T, cardinality)

    ei = ~(typemax(T) << N)
    for i in 1:cardinality
        basis[i] += ei
        ei = next_basis_element(ei)
    end

    basis
end

function build_HN(L, N)
    basis = build_basis_N(UInt32, L, N)
    d = length(basis)
    HN = zeros(d, d)

    index_map = Dict(basis .=> 1:d)

    for (b, n) in enumerate(basis)
        output = apply_H(n, L)
        for (m, h) in output
            a = index_map[m]
            HN[a, b] += h
        end
    end

    HN
end

#======================================================#
# Building HNk
#======================================================#

function build_basis_Nk(T::K, L, N, k) where K <: DataType
    basis = Tuple{T, Int64}[]
    basis_N = build_basis_N(T, L, N)

    for n in basis_N
        m, p = representative_state(n, L)
        if (n == m) && ((k * p) % L == 0)
            push!(basis, (n, p))
        end
    end
    basis
end

function representative_state(n::Unsigned, L)
    # This function assumes there are no bits beyond the L-th bit.
    # If there is one this will go into an infinite loop.
    rs = n
    d = 0
    m = translate(n, L)
    p = 1
    while m != n
        m < rs && (rs = m; d = p)
        p += 1
        m = translate(m, L)
    end
    rs, p, L - d
end

function translate(n::T, L)::T where T <: Union{UInt32, UInt64}
    m = n << 1
    m |= (n >> (L-1))
    m & ~(typemax(T) << L)
end

function build_HNk(L, N, k)
    basis = build_basis_Nk(UInt32, L, N, k)
    d = length(basis)
    HNk = zeros(d, d)
    index_map = Dict(e => i for (i, (e, p)) in enumerate(basis))
    ωk = cispi(2 * k / L)

    for (b, (n, pn)) in enumerate(basis)
        output = apply_H(n, L)
        YnL = √pn
        for (m, h) in output
            m_rs, pm, d = representative_state(m, L)
            a = index_map[m_rs]
            YmL = (√pm)
            HNk[a, b] += (YnL/YmL) * ωk^d * h
        end
    end

    HNk
end

#======================================================#
# Maximum Symmetry Sector
#======================================================#

function build_HMSS(L)
    basis = build_MSS_basis(L)
    d = length(basis)
    HMSS = zeros(d, d)
    index_map = Dict(e => i for (i, (e, qp)) in enumerate(basis))

    for (b, (n, qpn)) in enumerate(basis)
        output = apply_H(n, L)
        Zn4L = √qpn
        for (m, h) in output
            m_srs, qpm = super_representative_state(m, L)
            a = index_map[m_srs]
            Zm4L = √qpm
            HMSS[a, b] += (Zn4L/Zm4L) * h
        end
    end

    HMSS
end

function build_MSS_basis(L)
    basis = []
    basisNk = build_basis_Nk(UInt32, L, L÷2, 0)

    for (n, p) in basisNk
        n, nx, nr, nrx = related_representative_states(n, L)
        if n <= min(nx, nr, nrx)
            q = length(unique((n, nx, nr, nrx)))
            push!(basis, (n, q*p))
        end
    end

    basis
end

function super_representative_state(n, L)
    n, nx, nr, nrx = related_representative_states(n, L)
    n_srs = min(n, nx, nr, nrx)
    n_srs, length(unique((n, nx, nr, nrx)))
end

function related_representative_states(n, L)
    nx = invert(n, L)
    nr = reflect(n, L)
    nrx = reflect(invert(n, L), L)

    n,   p,   d   = representative_state(n, L)
    nx,  px,  dx  = representative_state(nx, L)
    nr,  pr,  dr  = representative_state(nr, L)
    nrx, prx, drx = representative_state(nrx, L)
    n, nx, nr, nrx
end

function invert(n::T, L) where T <: Unsigned
    mask = ~(typemax(T) << L)
    mask ⊻ n
end

function reflect(n::T, L) where T <: Unsigned
    nr = zero(T)
    i = 0
    while i < L
        nr <<= 1
        nr |= n & one(T)
        n >>= 1
        i += 1
    end
    nr
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