struct Solution
    cost::Float64
    clusters::Vector{Vector{Int}}
end

const min_viol = 1e-3
const max_nb_cuts = 5000
const target_nb_cuts = 1000

struct TriangleCut
    i::Int
    j::Int
    l::Int
    viol::Float64
end

function separate_triangle_cuts!(n::Int, z_::Matrix{Float64}, cuts::Vector{TriangleCut})::Vector{TriangleCut}
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    resize!(cuts, 0)
    for i in 1:n
        z_ii = zs_(i, i)
        for j in 1:n
            z_ij = zs_(i, j)
            if j == i || z_ij < min_viol
                continue
            end
            for l in (j+1):n
                z_il = zs_(i, l)
                if l == i || z_il < min_viol
                    continue
                end
                viol = z_ij + z_il - z_ii - zs_(j, l)
                if viol >= min_viol
                    push!(cuts, TriangleCut(i, j, l, viol))
                end
            end
        end
    end
    sort!(cuts, by = x -> x.viol, rev = true)
    return cuts
end

struct PivotCut
    i::Int
    j::Int
    viol::Float64
end

function separate_pivot_cuts!(n::Int, z_::Matrix{Float64}, cuts::Vector{PivotCut})::Vector{PivotCut}
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    resize!(cuts, 0)
    for i in 1:n
        z_ii = zs_(i, i)
        for j in 1:n
            z_ij = zs_(i, j)
            if j == i || z_ij < min_viol
                continue
            end
            viol = z_ij - z_ii
            if viol >= min_viol
                push!(cuts, PivotCut(i, j, viol))
            end
        end
    end
    sort!(cuts, by = x -> x.viol, rev = true)
    return cuts
end

struct SdpCut
    vec::Vector{Float64}
end

function compute_and_check_solution(
    data::Data{Dim},
    y_::Matrix{Float64},
    check_cost::Float64,
)::Solution where {Dim}
    # Find the clusters
    n = length(data.points)
    point_to_cluster = zeros(Int, n)
    point_to_cluster[n] = 1
    K = 1
    for i in (n-1):-1:1
        for j in (i+1):n
            if y_[i, j] > 0.5 * y_[i, i]
                point_to_cluster[i] = point_to_cluster[j]
                break
            end
        end
        if point_to_cluster[i] == 0
            K += 1
            point_to_cluster[i] = K
        end
    end

    # Find the centroids
    centroids = [ntuple(x -> 0.0, Dim) for _ in 1:K]
    cluster_sizes = zeros(Int, K)
    for i in 1:n
        k = point_to_cluster[i]
        cluster_sizes[k] += 1
        centroids[k] = centroids[k] .+ data.points[i]
    end
    for k in 1:K
        centroids[k] = centroids[k] ./ cluster_sizes[k]
    end

    # Compute the solution cost and check
    cost = 0.0
    for i in 1:n
        cost += sum((data.points[i] .- centroids[point_to_cluster[i]]) .^ 2)
    end
    if abs(cost - check_cost) > 1e-6 * max(cost, check_cost)
        println("ERROR: computed cost $cost does not match check cost $check_cost")
    end

    # Return the solution
    return Solution(cost, [findall(x -> x == k, point_to_cluster) for k in 1:K])
end

function solve(data::Data{Dim}, K::Int)::Solution where {Dim}
    println("Solving $(length(data.points)) points in $(Dim) dimensions")
    mat"maxNumCompThreads(1)"

    # Set the constants
    n = length(data.points)

    # Build the MIP model
    model = Model(SDPSolver.Optimizer)
    set_optimizer_attribute(model, QuietParam, QuietValue)
    @variables(model, begin
        z[i = 1:n, j = 1:n] >= ((i == j) ? (K / (n - K + 1)) : 0.0), PSD
    end)
    @objective(
        model,
        Min,
        -(K / n) * sum(data.costs[i, j] * z[i, j] for i in 1:n, j in 1:n)
    )
    @constraints(model, begin
        c1, sum(z[i, i] for i in 1:n) == n
        c2[i = 1:n], sum(z[i, j] for j in 1:n) == (n / K)
        # c3[i = 1:n, j = 1:n; i != j], z[i, i] >= z[i, j]
    end)

    # loop adding triangle cuts
    z_target = zeros(Float64, 0, 0)
    alpha = 0.0 # 0.99
    z_ = zeros(n, n)
    z_aux = zeros(n, n)
    pivot_cuts = Vector{PivotCut}()
    triangle_cuts = Vector{TriangleCut}()
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    function update_z_aux()
        for i in 1:n, j in 1:n
            z_aux[i, j] = alpha * z_target[i, j] + (1 - alpha) * z_[i, j]
        end
        return nothing
    end
    added_cuts = Vector{ConstraintRef}()
    should_keep = Vector{Bool}()
    cut_round = 0
    while true
        optimize!(model)
        obj = objective_value(model)
        remain_cuts = length(added_cuts)
        for c in 1:remain_cuts
            # should_keep[c] = get_attribute(added_cuts[c], MOI.ConstraintBasisStatus()) == MOI.BASIC
            should_keep[c] = abs(value(added_cuts[c])) < min_viol
            # should_keep[c] = abs(dual(added_cuts[c])) > 1e-6
            # should_keep[c] = true
        end
        c = 1
        while c <= remain_cuts
            if !should_keep[c]
                JuMP.delete(model, added_cuts[c])
                added_cuts[c], added_cuts[end] = added_cuts[end], added_cuts[c]
                pop!(added_cuts)
                should_keep[c], should_keep[end] = should_keep[end], should_keep[c]
                pop!(should_keep)
                remain_cuts -= 1
            end
            c += 1
        end
        optimize!(model)
        new_obj = objective_value(model)
        nb_infeas = 0
        max_infeas = 0.0
        for i in 1:n, j in i:n
            z_[i, j] = value(z[i, j])
            infeas = 0.5 - abs(z_[i, j] / z_[i, i] - 0.5)
            max_infeas = max(max_infeas, infeas)
            if infeas < 1e-2
                nb_infeas += 1
            end
        end
        if nb_infeas == 0
            break
        end
        if isempty(z_target)
            z_target = copy(z_)
        end

        cut_round += 1
        nb_cuts = 0
        first = true
        while first || (alpha > 0.0 && nb_cuts == 0)
            update_z_aux()
            nb_cuts = 0
            separate_pivot_cuts!(n, z_aux, pivot_cuts)
            resize!(pivot_cuts, min(max_nb_cuts, length(pivot_cuts)))
            for cut in pivot_cuts
                if zs_(cut.i, cut.i) >= zs_(cut.i, cut.j) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(added_cuts, @constraint(model, z[cut.i, cut.i] >= z[cut.i, cut.j]))
                if nb_cuts >= target_nb_cuts
                    break
                end
            end
            separate_triangle_cuts!(n, z_aux, triangle_cuts)
            resize!(triangle_cuts, min(max_nb_cuts, length(triangle_cuts)))
            for cut in triangle_cuts
                if zs_(cut.j, cut.l) >= zs_(cut.i, cut.j) + zs_(cut.i, cut.l) - zs_(cut.i, cut.i) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(
                    added_cuts,
                    @constraint(model, z[cut.j, cut.l] >= z[cut.i, cut.j] + z[cut.i, cut.l] - z[cut.i, cut.i])
                )
                if nb_cuts > 2 * target_nb_cuts
                    break
                end
            end
            if alpha != 0.0
                if nb_cuts < 2 * n
                    alpha -= (1 - alpha) / 3
                end
                if nb_cuts > 2 * n
                    alpha += (1 - alpha) / 5
                end
                if alpha < 0.95
                    alpha = 0.0
                end
            end
            first = false
        end
        resize!(should_keep, length(added_cuts))

        diff = round(obj - new_obj, digits = 5)
        new_obj = data.fixed_cost + new_obj
        @show cut_round, new_obj, diff, nb_cuts, remain_cuts, alpha
        if nb_cuts == 0
            break
        end
    end

    # return the result
    # println("z_ = $z_")
    # @show objective_value(model)
    # @show -(K / n) * 1e-3 * sum(data.costs[i, j] * z_[i, j] for i in 1:n, j in 1:n)
    return compute_and_check_solution(data, z_, data.fixed_cost + objective_value(model))
end
