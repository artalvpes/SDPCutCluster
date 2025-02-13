struct Data{DIM}
    fixed_cost::Float64
    points::Vector{NTuple{DIM, Float64}}
    dists::Matrix{Float64}
    costs::Matrix{Float64}
end

function read_data(filename::String)::Data
    fixed_cost = 0.0
    dim = 0
    coords = Vector{Vector{Float64}}()
    open(filename) do file
        nb_points = 0
        for line in eachline(file)
            if dim == 0
                head = parse.(Int, split(line))
                nb_points, dim = head[1], head[2]
            else
                push!(coords, map(x -> parse(Float64, x), split(line)))
            end
        end
    end
    centroid = [sum(coords[i][j] for i in 1:length(coords)) / length(coords) for j in 1:dim]
    coords = [coords[i] .- centroid for i in 1:length(coords)]
    points = [ntuple(x -> p[x], dim) for p in coords]
    dists = [sum((pi .- pj) .^ 2) for pi in points, pj in points]
    costs = [sum((pi .* pj)) for pi in points, pj in points]
    fixed_cost = sum([costs[i, i] for i in 1:length(points)])
    return Data{dim}(fixed_cost, points, dists, costs)
end
