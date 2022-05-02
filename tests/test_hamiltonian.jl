using DataFrames
using DrWatson
using Test
# using GLMakie, Makie
using SparseArrays
using FileIO
using LinearAlgebra

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

    # df = collect_results(datadir("test_data"))

    # for r in eachrow(df)
    #     H_old = r.H
    #     @unpack L, N, J1, V1, J2, V2, hs, is = r.params

    #     H_cur = XXZ.build_matrix_N(h, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is)
    #     @test H_old == H_cur
    # end
end


#===========================#
# For manually checking the Hamiltonian matrix:

# L = 4; N = L÷2
# J1 = 0.0; V1 = 0.0
# J2 = 0.0; V2 = 1.0
# pbc = false
# hs = (0.0,); is = (1)

# H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)

#===========================#

T1_obc = [0 1 0 0 0 0;
          1 0 1 1 0 0;
          0 1 0 0 1 0;
          0 1 0 0 1 0;
          0 0 1 1 0 1;
          0 0 0 0 1 0]

T1_pbc = [0 1 0 0 1 0;
          1 0 1 1 0 1;
          0 1 0 0 1 0;
          0 1 0 0 1 0;
          1 0 1 1 0 1;
          0 1 0 0 1 0]

T2_obc = [0 0 1 1 0 0;
          0 0 0 0 0 0;
          1 0 0 0 0 1;
          1 0 0 0 0 1;
          0 0 0 0 0 0;
          0 0 1 1 0 0]

T2_pbc = [0 0 2 2 0 0;
          0 0 0 0 0 0;
          2 0 0 0 0 2;
          2 0 0 0 0 2;
          0 0 0 0 0 0;
          0 0 2 2 0 0]

V1_pbc = [0, -4, 0, 0, -4, 0] |> diagm
V1_obc = [1, -3, -1, -1, -3, 1] |> diagm
V2_pbc = [-4, 4, -4, -4, 4, -4] |> diagm
V2_obc = [-2, 2, -2, -2, 2, -2] |> diagm

h1 = [1, -1, 1, -1, 1, -1] |> diagm
h2 = [-1, 1, 1, -1, -1, 1] |> diagm

@testset "Check Hamiltonian" begin
    L = 4; N = L÷2
    J1 = 0.0; V1 = 0.0
    J2 = 0.0; V2 = 0.0
    hs = (0.0,); is = (1)

    # Check if the correct kinetic terms are created.
    t = 1.0; pbc = false
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=t, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == T1_obc

    t = 1.0; pbc = true
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=t, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == T1_pbc

    t = 1.0; pbc = false
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=t, V2=V2, hs=hs, is=is, pbc)
    @test H == T2_obc

    t = 1.0; pbc = true
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=t, V2=V2, hs=hs, is=is, pbc)
    @test H == T2_pbc



    # Check if the correct interaction terms are created.
    t = 1.0; pbc = false
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=t, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == V1_obc

    t = 1.0; pbc = true
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=t, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == V1_pbc

    t = 1.0; pbc = false
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=t, hs=hs, is=is, pbc)
    @test H == V2_obc

    t = 1.0; pbc = true
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=t, hs=hs, is=is, pbc)
    @test H == V2_pbc



    # Check if the correct impurity terms are created.
    hs = (1.0,); is = (1); pbc = false
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == h1

    hs = (1.0,); is = (2); pbc = true
    H = XXZ.build_matrix_N(XXZ.apply_H, L, N; J1=J1, V1=V1, J2=J2, V2=V2, hs=hs, is=is, pbc)
    @test H == h2
end