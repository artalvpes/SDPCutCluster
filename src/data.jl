struct Data{DIM}
    fixed_cost::Float64
    points::Vector{NTuple{DIM, Float64}}
    dists::Matrix{Float64}
    costs::Matrix{Float64}
end

default_zero(val::Union{Float64, Nothing}) = isnothing(val) ? 0.0 : val

function Data{DIM}(filename::String, ignore::Int) where {DIM}
    fixed_cost = 0.0
    points = Vector{NTuple{DIM, Float64}}()
    open(filename) do file
        for line in eachline(file)
            if isempty(line) || !(line[1] in '0':'9') && line[1] != '-'
                continue
            end
            coords = map(x -> default_zero(tryparse(Float64, x)), split(line, ','))[(ignore+1):end]
            push!(points, ntuple(x -> coords[x], DIM))
        end
    end
    dists = [sum((pi .- pj) .^ 2) for pi in points, pj in points]
    costs = [sum((pi .* pj)) for pi in points, pj in points]
    fixed_cost = sum([costs[i, i] for i in 1:length(points)])
    return Data{DIM}(fixed_cost, points, dists, costs)
end
