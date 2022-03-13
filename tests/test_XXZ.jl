using DrWatson
using Random
using Test

include(srcdir("XXZ_model.jl"))
include(srcdir("XXZ_model_naive.jl"))

for L in [28, 32]
    for i = 1:100
        b = bitrand(L)
        # println(b)

        output1 = apply_H(b)
        output2 = apply_H(UInt32(b.chunks[1]), L)

        @test length(output1) == length(output2)
        for i = 1:length(output1)
            bi, wi = output1[i]
            ni, ωi = output2[i]
            @test ni == UInt32(bi.chunks[1])
            @test ωi == wi
        end
    end
end

i = 1
for (m, weight) in apply_H(UInt32(b.chunks[1]), 28)
    println(bitstring(m), " ", weight, " ", i)
    i += 1
end

i = 1
for (m, weight) in apply_H(b)
    println(bitstring(UInt32(m.chunks[1])), " ", weight, " ", i)
    i += 1
end