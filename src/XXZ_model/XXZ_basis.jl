#======================================================#
# Particle Number Sector
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

Assumes current basis element is not zero.
"""
function next_basis_element(e::U)::U where U <: Unsigned
    c = e & -e
    r = e + c
    (((r ⊻ e) >> 2) ÷ c) | r
end

"""
Returns a vector of particle-basis elements for the XXZ model of length `L` 
with `N` particles.
"""
function build_basis_N(T::K, L::Integer, N::Integer) where K <: DataType
    cardinality = binomial(L, N)
    basis = zeros(T, cardinality)

    ei = ~(typemax(T) << N)
    i = 1
    while true
        basis[i] += ei
        i == cardinality && break
        ei = next_basis_element(ei)
        i += 1
    end

    basis
end


#======================================================#
# Momentum Sector
#======================================================#

"""
Returns a vector of basis elements for the number-momentum sector
N, k of an XXZ model of length L.
"""
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

"""
Returns the representative state for the equivalence class n is a member of.
The period and distance from the representative state to n is also returned.
"""
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


#======================================================#
# Maximum Symmetry Sector
#======================================================#

"""
Return a vector of basis elements for the maximum symmetry sector
of an XXZ model with length L. L is assumed to be even.
"""
function build_MSS_basis(L)
    basis = []
    basisNk = build_basis_Nk(UInt32, L, L÷2, 0)

    for (n, p) in basisNk
        n, nx, nr, nrx, _ = related_representative_states(n, L)
        if n <= min(nx, nr, nrx)
            q = length(unique((n, nx, nr, nrx)))
            push!(basis, (n, q*p))
        end
    end

    basis
end

"""
Return the representative states for the equivalence classes related to
the equivalence class containing n by inversion and reflection transformations.
"""
function related_representative_states(n, L)
    nx = invert(n, L)
    nr = reverse(n, L)
    nrx = reverse(invert(n, L), L)

    n,   p,   d   = representative_state(n, L)
    nx,  px,  dx  = representative_state(nx, L)
    nr,  pr,  dr  = representative_state(nr, L)
    nrx, prx, drx = representative_state(nrx, L)
    n, nx, nr, nrx, p
end

function MSS_to_full_basis(n::T, L) where T <: Unsigned
    ns = zeros(T, L); ns[1] = n
    nxs = zeros(T, L); nxs[1] = invert(n, L)
    nrs = zeros(T, L); nrs[1] = reverse(n, L)
    nrxs = zeros(T, L); nrxs[1] = reverse(invert(n, L), L)

    for i = 2:L
        ns[i] = translate(ns[i-1], L)
        nxs[i] = translate(nxs[i-1], L)
        nrs[i] = translate(nrs[i-1], L)
        nrxs[i] = translate(nrxs[i-1], L)
    end

    [ns; nxs; nrs; nrxs]
end