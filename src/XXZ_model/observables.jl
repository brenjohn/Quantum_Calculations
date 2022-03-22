###
### Zero momentum distribution.
###

"""
Returns the image of n under the zero momentum distribution operator.
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

###
### Nearest neigbour interaction energy density.
###

"""
Returns the image of n under the nearest neigbour interaction energy density operator.
"""
function apply_B(n::Unsigned, L)
    nn = translate(n, L)
    nn = n & nn
    weight = hamming_weight(nn) / L
    [(n, weight)]
end

function build_matrix_MSS(operator::Function, L)
    basis = build_MSS_basis(L)
    d = length(basis)
    M = zeros(d, d)
    index_map = Dict(e => i for (i, (e, qp)) in enumerate(basis))

    for (b, (n, qpn)) in enumerate(basis)
        output = operator(n, L)
        Zn4L = √qpn
        for (m, h) in output
            m_srs, qpm = super_representative_state(m, L)
            a = index_map[m_srs]
            Zm4L = √qpm
            M[a, b] += (Zn4L/Zm4L) * h
        end
    end

    M
end