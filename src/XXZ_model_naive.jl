function apply_H(n::BitVector)
    L = length(n)
    Δ = 2
    λ = 0.2

    diag = -length(n) + 2 * (n[1] == n[end])
    for l = 1:length(n) - 1
        diag += 2 * (n[l] == n[l+1])
    end
    diag *= (-Δ/2)

    if λ > 0
        diag2 = -length(n) + 2 * (n[1] == n[end-1]) + 2 * (n[2] == n[end])
        for l = 1:length(n) - 2
            diag2 += 2 * (n[l] == n[l+2])
        end
        diag2 *= (-Δ/2)
        diag = diag/(1+λ) + λ * diag2
    end

    output = [(n, diag)]
    for l = 1:L
        if n[l] != n[(l%L+1)]
            m = bitflip(n, l, l%L + 1)
            push!(output, (m, -1))
        end
    end

    if λ > 0
        for l = 1:L
            if n[l] != n[(l+2-1)%L + 1]
                m = bitflip(n, l, (l+2-1)%L + 1)
                push!(output, (m, -λ))
            end
        end
    end

    output
end

function bitflip(n::BitVector, i, j)
    m = similar(n)
    m.chunks[1] = n.chunks[1]
    m[i] = !m[i]
    m[j] = !m[j]
    m
end