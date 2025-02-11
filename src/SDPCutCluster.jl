module SDPCutCluster

using JuMP, CPLEX, MathOptInterface, SDPNAL, LinearAlgebra, MATLAB

const SDPSolver = SDPNAL

include("data.jl")
include("solver.jl")

function main(args::Vector{String})
    println("SDP-and-Cut Solver for Clustering Problems")
    if length(args) < 3
        println("Usage: julia src/run.jl <instance> <dimension> <#clusters> [<#cols to ignore>]")
        return
    end
    filename = args[1]
    dim = parse(Int, args[2])
    num_clusters = parse(Int, args[3])
    ignore = 0
    if length(args) > 3
        ignore = parse(Int, args[4])
    end
    data = Data{dim}("$filename", ignore)
    cputime = @elapsed sol = solve(data, num_clusters)
    @show cputime
    println("Cost: $(sol.cost)")
    println("Clusters: $(sol.clusters)")
end

export main

end # module ColGenCluster
