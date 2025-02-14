struct Solution
    cost::Float64
    clusters::Vector{Vector{Int}}
end

const target_tol = 1e-6
const tol_step = 0.5
const max_nb_cuts = 5000
const target_nb_cuts = 1000

tol_is_ok(tol::Float64)::Bool = abs(tol - target_tol) < 1e-6 * target_tol

struct TriangleCut
    i::Int
    j::Int
    l::Int
    viol::Float64
end

function separate_triangle_cuts!(
    n::Int,
    z_::Matrix{Float64},
    cuts::Vector{TriangleCut},
    min_viol::Float64,
)::Vector{TriangleCut}
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

function separate_pivot_cuts!(
    n::Int,
    z_::Matrix{Float64},
    cuts::Vector{PivotCut},
    min_viol::Float64,
)::Vector{PivotCut}
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

function assign_to_clusters(
    data::Data{Dim},
    centroids::Vector{NTuple{Dim, Float64}},
    points_to_cluster::Vector{Int},
    cluster_sizes::Vector{Int},
)::Nothing where {Dim}
    n = length(data.points)
    K = length(cluster_sizes)
    resize!(centroids, 2 * K)
    for k in 1:K
        centroids[K+k] = ntuple(_ -> 0.0, Dim)
    end
    fill!(cluster_sizes, 0)
    for i in 1:n
        min_dist = Inf
        for k in 1:K
            dist = sum((data.points[i] .- centroids[k]) .^ 2)
            if dist < min_dist
                min_dist = dist
                points_to_cluster[i] = k
            end
        end
        centroids[K+points_to_cluster[i]] = centroids[K+points_to_cluster[i]] .+ data.points[i]
        cluster_sizes[points_to_cluster[i]] += 1
    end
    for k in 1:K
        centroids[k] = centroids[K+k] ./ cluster_sizes[k]
    end
    resize!(centroids, K)
    return nothing
end

function run_k_means(
    data::Data{Dim},
    z_sol::Matrix{Float64},
    sol_cost::Float64,
    centroids::Vector{NTuple{Dim, Float64}},
    points_to_cluster::Vector{Int},
    cluster_sizes::Vector{Int},
)::Float64 where {Dim}
    n = length(data.points)
    cost = sum(sum((data.points[i] .- centroids[points_to_cluster[i]]) .^ 2) for i in 1:n)
    # @show cost
    while true
        assign_to_clusters(data, centroids, points_to_cluster, cluster_sizes)
        new_cost = sum(sum((data.points[i] .- centroids[points_to_cluster[i]]) .^ 2) for i in 1:n)
        if new_cost >= cost - 1e-6 * max(cost, new_cost)
            cost = new_cost
            break
        end
        @show new_cost
        # cost = new_cost
    end
    if sol_cost > cost
        for i in 1:n, j in 1:n
            k = points_to_cluster[i]
            z_sol[i, j] = (k == points_to_cluster[j]) ? (1 / cluster_sizes[k]) : 0.0
        end
        return cost
    end
    return sol_cost
end

function run_rounding_heuristic(
    data::Data{Dim},
    z_::Matrix{Float64},
    z_sol::Matrix{Float64},
    sol_cost::Float64,
    centroids::Vector{NTuple{Dim, Float64}},
    points_to_cluster::Vector{Int},
    cluster_sizes::Vector{Int},
    # min_dist::Vector{Float64},
    # closest::Vector{Int},
    unused::Vector{Int},
)::Float64 where {Dim}
    n = length(data.points)
    K = length(cluster_sizes)
    resize!(unused, n)
    for i in 1:n
        unused[i] = i
    end
    for k in 1:K
        pos = rand(1:length(unused))
        unused[pos], unused[end] = unused[end], unused[pos]
        i = pop!(unused)
        centroids[k] = data.points[i]
        sort!(unused, by = j -> z_[i, j])
        s = min(length(unused) - K + k, ceil(Int, (n / K) * (1 / z_[i, i])))
        first = length(unused) + 2 - s
        for pos in first:length(unused)
            centroids[k] = centroids[k] .+ data.points[unused[pos]]
        end
        centroids[k] = centroids[k] ./ s
        resize!(unused, first - 1)
    end
    assign_to_clusters(data, centroids, points_to_cluster, cluster_sizes)

    # # function to compute a distance weighted by the correspondence between the leaders given by the
    # # SDP relaxation
    # dist_(i::Int, j::Int) =
    #     sum((centroids[i] .- centroids[j]) .^ 2) * (1.1 - z_[i, j] / sqrt(z_[i, i] * z_[j, j]))

    # # initializations
    # n = length(data.points)
    # K = length(cluster_sizes)
    # for i in 1:n
    #     points_to_cluster[i] = i
    # end
    # resize!(centroids, n)
    # centroids .= data.points
    # resize!(cluster_sizes, n)
    # fill!(cluster_sizes, 1)
    # fill!(min_dist, Inf)
    # for i in 1:n, j in (i+1):n
    #     d_ = dist_(i, j)
    #     if min_dist[i] > d_
    #         min_dist[i] = d_
    #         closest[i] = j
    #     end
    # end

    # # loop joining clusters until only K remain
    # nb_clusters = n
    # while nb_clusters > K
    #     # find the smallest cluster distance to join
    #     to_join = 0
    #     join_dist = Inf
    #     for i in 1:n
    #         if points_to_cluster[i] != i
    #             continue
    #         end
    #         if join_dist > min_dist[i]
    #             join_dist = min_dist[i]
    #             to_join = i
    #         end
    #     end
    #     i = to_join
    #     j = closest[i]

    #     # join clusters i and j choosing the leader randomly
    #     if rand(Bool)
    #         points_to_cluster[j] = i
    #     else
    #         points_to_cluster[i] = j
    #         i, j = j, i
    #     end

    #     # update the centroids and distances
    #     centroids[i] =
    #         (
    #             cluster_sizes[i] .* centroids[i] .+ cluster_sizes[j] .* centroids[j]
    #         ) ./ (cluster_sizes[i] + cluster_sizes[j])
    #     cluster_sizes[i] += cluster_sizes[j]
    #     min_dist[i] = Inf
    #     for l in 1:n
    #         if points_to_cluster[l] != l || l == i
    #             continue
    #         end
    #         d_ = dist_(i, l)
    #         if i < l
    #             if min_dist[i] > d_
    #                 min_dist[i] = d_
    #                 closest[i] = l
    #             end
    #         else
    #             if min_dist[l] > d_
    #                 min_dist[l] = d_
    #                 closest[l] = i
    #             end
    #         end
    #     end
    #     for i in 1:n
    #         if closest[i] == j
    #             min_dist[i] = Inf
    #             for l in (i+1):n
    #                 if points_to_cluster[l] != l
    #                     continue
    #                 end
    #                 d_ = dist_(i, l)
    #                 if min_dist[i] > d_
    #                     min_dist[i] = d_
    #                     closest[i] = l
    #                 end
    #             end
    #         end
    #     end
    #     nb_clusters -= 1
    # end

    # # compact the clusters vectors and run the K-means to improve
    # k = 0
    # for i in 1:n
    #     if points_to_cluster[i] == i
    #         k += 1
    #         points_to_cluster[i] = k
    #         centroids[k] = centroids[i]
    #         cluster_sizes[k] = cluster_sizes[i]
    #     else
    #         points_to_cluster[i] = -points_to_cluster[i]
    #     end
    # end
    # changed = true
    # while changed
    #     changed = false
    #     for i in 1:n
    #         if points_to_cluster[i] < 0
    #             points_to_cluster[i] = points_to_cluster[-points_to_cluster[i]]
    #             changed = true
    #         end
    #     end
    # end
    # resize!(centroids, K)
    # resize!(cluster_sizes, K)
    return run_k_means(data, z_sol, sol_cost, centroids, points_to_cluster, cluster_sizes)
end

struct HeuristicBuffers{Dim}
    centroids::Vector{NTuple{Dim, Float64}}
    points_to_cluster::Vector{Int}
    cluster_sizes::Vector{Int}
    # min_dist::Vector{Float64}
    # closest::Vector{Int}
    unused::Vector{Int}
end

function HeuristicBuffers{Dim}(n::Int, K::Int) where {Dim}
    return HeuristicBuffers{Dim}(
        Vector{NTuple{Dim, Float64}}(undef, K),
        Vector{Int}(undef, n),
        Vector{Int}(undef, K),
        # Vector{Float64}(undef, n),
        Vector{Int}(undef, n),
    )
end

function compute_and_check_solution(
    data::Data{Dim},
    y_::Matrix{Float64},
    check_cost::Float64,
    check_K::Int,
)::Solution where {Dim}
    # Find the clusters
    n = length(data.points)
    points_to_cluster = zeros(Int, n)
    points_to_cluster[n] = 1
    K = 1
    for i in (n-1):-1:1
        for j in (i+1):n
            if y_[i, j] > 0.5 * y_[i, i]
                points_to_cluster[i] = points_to_cluster[j]
                break
            end
        end
        if points_to_cluster[i] == 0
            K += 1
            points_to_cluster[i] = K
        end
    end

    # Find the centroids
    centroids = [ntuple(x -> 0.0, Dim) for _ in 1:K]
    cluster_sizes = zeros(Int, K)
    for i in 1:n
        k = points_to_cluster[i]
        cluster_sizes[k] += 1
        centroids[k] = centroids[k] .+ data.points[i]
    end
    for k in 1:K
        centroids[k] = centroids[k] ./ cluster_sizes[k]
    end

    # Compute the solution cost and check
    cost = 0.0
    for i in 1:n
        cost += sum((data.points[i] .- centroids[points_to_cluster[i]]) .^ 2)
    end
    if K != check_K
        println("ERROR: solution has $K clusters instead of $check_K")
    end
    if abs(cost - check_cost) > 1e-6 * max(cost, check_cost)
        println("ERROR: computed cost $cost does not match check cost $check_cost")
    end

    # Return the solution
    return Solution(cost, [findall(x -> x == k, points_to_cluster) for k in 1:K])
end

function solve(data::Data{Dim}, K::Int)::Solution where {Dim}
    println("Solving $(length(data.points)) points in $(Dim) dimensions")
    mat"maxNumCompThreads(1)"

    # Set the constants
    n = length(data.points)

    # Build the MIP model
    model = Model(SDPSolver.Optimizer)
    if SDPSolverName == "SDPNAL"
        set_optimizer_attribute(model, "printlevel", 0)
        set_optimizer_attribute(model, "stopoption", 0)
    else
        set_optimizer_attribute(model, "verbose", false)
    end
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
    target_obj = Inf
    z_target = zeros(Float64, n, n)
    buffers = HeuristicBuffers{Dim}(n, K)
    alpha = 0.9
    z_ = zeros(Float64, n, n)
    z_aux = zeros(Float64, n, n)
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
    obj = 0.0
    curr_tol = 1e-2
    min_viol = sqrt(curr_tol)
    tol_was_decreased = false
    while true
        # solve the SDP relaxation
        if SDPSolverName == "SDPNAL"
            set_optimizer_attribute(model, "tol", curr_tol)
        else
            set_optimizer_attribute(model, "eps_rel", curr_tol)
        end
        sdp_time = @elapsed optimize!(model)
        new_obj = data.fixed_cost + objective_value(model)
        remain_cuts = length(added_cuts)
        for c in 1:remain_cuts
            # should_keep[c] = get_attribute(added_cuts[c], MOI.ConstraintBasisStatus()) == MOI.BASIC
            # should_keep[c] = abs(value(added_cuts[c])) < min_viol
            should_keep[c] = abs(dual(added_cuts[c])) > 1e-6 * new_obj
            # should_keep[c] = true
        end
        nb_infeas = 0
        max_infeas = 0.0
        for i in 1:n, j in 1:n
            z_[i, j] = value(z[i, j])
            # d_[i, j] = dual(LowerBoundRef(z[i, j]))
            infeas = 0.5 - abs(z_[i, j] / z_[i, i] - 0.5)
            max_infeas = max(max_infeas, infeas)
            if infeas < 1e-2
                nb_infeas += 1
            end
        end
        if nb_infeas == 0
            break
        end
        target_obj = run_rounding_heuristic(
            data,
            z_,
            z_target,
            target_obj,
            buffers.centroids,
            buffers.points_to_cluster,
            buffers.cluster_sizes,
            # buffers.min_dist,
            # buffers.closest,
            buffers.unused,
        )

        # remove cuts that are not needed
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

        # add new cuts
        cut_round += 1
        nb_cuts = 0
        first = true
        while first || (alpha > 0.0 && nb_cuts == 0)
            update_z_aux()
            nb_cuts = 0
            separate_pivot_cuts!(n, z_aux, pivot_cuts, min_viol)
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
            separate_triangle_cuts!(n, z_aux, triangle_cuts, min_viol)
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
                if alpha < 0.1
                    alpha = 0.0
                end
            end
            first = false
        end
        resize!(should_keep, length(added_cuts))

        diff = round(new_obj - obj, digits = 5)
        obj = new_obj
        @show cut_round, target_obj, new_obj, diff, nb_cuts, remain_cuts, alpha, curr_tol, sdp_time
        if nb_cuts == 0 || (new_obj > target_obj) || (diff < 1e-6 * new_obj && !tol_was_decreased)
            if tol_is_ok(curr_tol)
                break
            end
            curr_tol = max(curr_tol * tol_step, target_tol)
            min_viol = sqrt(curr_tol)
            tol_was_decreased = true
        else
            tol_was_decreased = false
        end
    end

    # return the result
    # println("z_ = $z_")
    # @show objective_value(model)
    # @show -(K / n) * 1e-3 * sum(data.costs[i, j] * z_[i, j] for i in 1:n, j in 1:n)
    return compute_and_check_solution(data, z_target, target_obj, K)
end
