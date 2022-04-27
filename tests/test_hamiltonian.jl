using DataFrames
using DrWatson
using Test
# using GLMakie, Makie
using SparseArrays
using FileIO

include(srcdir("XXZ_model/XXZ_model.jl"))
h = XXZ.apply_H

@testset "Check Hamiltonian" begin
    params = Dict(
        :L => [6, 8],
        :N => [3, 4],
        :J1 => 1,
        :V1 => 1,
        :J2 => [0, 1],
        :V2 => [0, 1],
        :hs => (),
        :is => (),
        :pbc=> true
    )

    # for ps in dict_list(params)
    #     @unpack L, N, J1, V1, J2, V2, hs, is = ps
    #     H = XXZ.build_matrix_N(h, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is)

    #     name = savename("Hamiltonian", ps, ".jld2")
    #     filename = joinpath(datadir("test_data"), name)
    #     safesave(filename, Dict("H" => H, "params" => ps))
    # end

    df = collect_results(datadir("test_data"))

    for r in eachrow(df)
        H_old = r.H
        @unpack L, N, J1, V1, J2, V2, hs, is = r.params

        H_cur = XXZ.build_matrix_N(h, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is)
        @test H_old == H_cur
    end
end
