#=============================================================#
# Utility functions for generating XXZ hamiltonians and basis #
#=============================================================#

"""
Return if the i-th and j-th bit of n are different. (Assumes i < j)
"""
function bits_differ(n::Unsigned, i, j)
    ((n & (1 << i)) << (j-i)) ⊻ (n & (1 << j)) != 0
end

"""
Returns an unsigned int equivalent to n but the i-th and j-th bits flipped.
"""
function flipbits(n::Unsigned, i, j)
    # This function assumes i < j
    m = ((1 << (j - i)) + 1) << i
    n ⊻ m
end

"""
Returns the unsigned int generated by flipping the first L bits of n.
"""
function invert(n::T, L) where T <: Unsigned
    mask = ~(typemax(T) << L)
    mask ⊻ n
end

"""
Returns the unsigned int generated by reversing the first L bits of n.
"""
function reverse(n::T, L) where T <: Unsigned
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

"""
Returns the unsigned int generated by shifting the first L bits of n one space,
with periodic boundary conditions.
"""
function translate(n::T, L)::T where T <: Union{UInt32, UInt64}
    m = n << 1
    m |= (n >> (L-1))
    m & ~(typemax(T) << L)
end

"""
Returns the number of aligned neigbours that are a distance i apart.
"""
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