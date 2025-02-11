module SDPCutCluster

using JuMP, CPLEX, MathOptInterface, SDPNAL, LinearAlgebra, MATLAB

const SDPSolver = SDPNAL

include("data.jl")
include("solver.jl")

function main(args::Vector{String})
    println("SDP-and-Cut Solver for Clustering Problems")
    if length(args) < 2
        println("Usage: julia src/run.jl <instance> <#clusters>")
        return
    end
    filename = args[1]
    num_clusters = parse(Int, args[2])
    data = read_data("$filename")
    cputime = @elapsed sol = solve(data, num_clusters)
    @show cputime
    println("Cost: $(sol.cost)")
    println("Clusters: $(sol.clusters)")
end

export main

end # module ColGenCluster
