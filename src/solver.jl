struct Solution
    cost::Float64
    clusters::Vector{Vector{Int}}
end

const init_tol = 1e-5
const target_tol = 1e-6
const ph1_to_ph2_tol = 10
const gap_tol = 1e-4
const tol_step = 0.1
const max_nb_cuts = 100000
const target_nb_cuts = 5000
const max_safe_bound_iters = 10

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
    perm::Vector{Int},
)::Vector{TriangleCut}
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    resize!(cuts, 0)
    for i in perm
        z_ii = zs_(i, i)
        for j in (i+1):n
            z_ij = zs_(i, j)
            z_jj = zs_(j, j)
            for l in (j+1):n
                z_il = zs_(i, l)
                z_jl = zs_(j, l)
                z_ll = zs_(l, l)
                viol = z_ij + z_il - z_ii - z_jl
                if viol >= min_viol
                    push!(cuts, TriangleCut(i, j, l, viol))
                end
                viol = z_ij + z_jl - z_jj - z_il
                if viol >= min_viol
                    push!(cuts, TriangleCut(j, i, l, viol))
                end
                viol = z_il + z_jl - z_ll - z_ij
                if viol >= min_viol
                    push!(cuts, TriangleCut(l, i, j, viol))
                end
                if length(cuts) >= max_nb_cuts
                    break
                end
            end
            if length(cuts) >= max_nb_cuts
                break
            end
        end
        if length(cuts) >= max_nb_cuts
            break
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
    perm::Vector{Int},
)::Vector{PivotCut}
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    resize!(cuts, 0)
    for i in perm
        z_ii = zs_(i, i)
        for j in (i+1):n
            z_ij = zs_(i, j)
            z_jj = zs_(j, j)
            viol = z_ij - z_ii
            if viol >= min_viol
                push!(cuts, PivotCut(i, j, viol))
            end
            viol = z_ij - z_jj
            if viol >= min_viol
                push!(cuts, PivotCut(j, i, viol))
            end
            if length(cuts) >= max_nb_cuts
                break
            end
        end
        if length(cuts) >= max_nb_cuts
            break
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
        # @show new_cost
        cost = new_cost
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
    rng::MersenneTwister,
    data::Data{Dim},
    z_::Matrix{Float64},
    z_sol::Matrix{Float64},
    sol_cost::Float64,
    centroids::Vector{NTuple{Dim, Float64}},
    points_to_cluster::Vector{Int},
    cluster_sizes::Vector{Int},
    unused::Vector{Int},
)::Float64 where {Dim}
    n = length(data.points)
    K = length(cluster_sizes)
    resize!(unused, n)
    for i in 1:n
        unused[i] = i
    end
    for k in 1:K
        pos = rand(rng, 1:length(unused))
        unused[pos], unused[end] = unused[end], unused[pos]
        i = pop!(unused)
        centroids[k] = data.points[i]
        sort!(unused, by = j -> z_[i, j])
        s = min(length(unused) - K + k, ceil(Int, 1 / z_[i, i]))
        first = length(unused) + 2 - s
        for pos in first:length(unused)
            centroids[k] = centroids[k] .+ data.points[unused[pos]]
        end
        centroids[k] = centroids[k] ./ s
        resize!(unused, first - 1)
    end
    assign_to_clusters(data, centroids, points_to_cluster, cluster_sizes)
    return run_k_means(data, z_sol, sol_cost, centroids, points_to_cluster, cluster_sizes)
end

struct HeuristicBuffers{Dim}
    centroids::Vector{NTuple{Dim, Float64}}
    points_to_cluster::Vector{Int}
    cluster_sizes::Vector{Int}
    unused::Vector{Int}
end

function HeuristicBuffers{Dim}(n::Int, K::Int) where {Dim}
    return HeuristicBuffers{Dim}(
        Vector{NTuple{Dim, Float64}}(undef, K),
        Vector{Int}(undef, n),
        Vector{Int}(undef, K),
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

function compute_P!(
    n::Int,
    K::Int,
    costs::Matrix{Float64},
    _pi::Float64,
    _sigma::Vector{Float64},
    _alpha::Matrix{Float64},
    cut_duals::Vector{Float64},
    cut_indices::Vector{Tuple{Int, Int, Int}},
    P::Matrix{Float64},
)
    for i in 1:n, j in 1:n
        P[i, j] = -costs[i, j] - _sigma[i] - _alpha[i, j]
        if i == j
            P[i, j] -= _pi
        end
    end
    for c in 1:length(cut_duals)
        i, j, l = cut_indices[c]
        if l == 0
            P[i, j] += cut_duals[c]
            P[i, i] -= cut_duals[c]
        else
            P[j, l] -= cut_duals[c]
            P[i, j] += cut_duals[c]
            P[i, l] += cut_duals[c]
            P[i, i] -= cut_duals[c]
        end
    end
    for i in 1:n, j in (i+1):n
        P[i, j] = P[j, i] = 0.5 * (P[i, j] + P[j, i])
    end
    return nothing
end

function compute_eigen_gradient!(
    cut_indices::Vector{Tuple{Int, Int, Int}},
    eigvec::Vector{Float64},
    grad::Vector{Float64},
)
    grad .= 0.0
    for c in 1:length(cut_indices)
        i, j, l = cut_indices[c]
        if l == 0
            grad[c] += eigvec[i] * eigvec[j] - eigvec[i] * eigvec[i]
        else
            grad[c] +=
                eigvec[i] * eigvec[j] + eigvec[i] * eigvec[l] - eigvec[i] * eigvec[i] - eigvec[j] * eigvec[l]
        end
    end
end

function compute_safe_bound(
    data::Data{Dim},
    _pi::Float64,
    _sigma::Vector{Float64},
    cut_duals::Vector{Float64},
    cut_indices::Vector{Tuple{Int, Int, Int}},
    _alpha::Matrix{Float64},
    K::Int,
    P::Matrix{Float64},
    cut_grad::Vector{Float64},
)::Float64 where {Dim}
    n = length(data.points)
    # write("fixed_cost.bin", data.fixed_cost)
    # write("costs.bin", data.costs)
    # write("pi.bin", _pi)
    # write("sigma.bin", _sigma)
    # write("cut_duals.bin", cut_duals)
    # write("cut_i.bin", getindex.(cut_indices, 1))
    # write("cut_j.bin", getindex.(cut_indices, 2))
    # write("cut_l.bin", getindex.(cut_indices, 3))
    # write("alpha.bin", _alpha)

    # compute the minimum eigenvalue of the P matrix and try to improve it by gradient descent
    compute_P!(n, K, data.costs, _pi, _sigma, _alpha, cut_duals, cut_indices, P)
    dec = eigen(P)
    lambda_min = dec.values[1]
    for _ in 1:max_safe_bound_iters
        eigvec = dec.vectors[:, 1]
        compute_eigen_gradient!(cut_indices, eigvec, cut_grad)
        norm_g = sqrt(sum(cut_grad .^ 2))
        cut_duals = map(x -> max(0, x), cut_duals .+ (-dec.values[1] / norm_g) * cut_grad / norm_g)
        compute_P!(n, K, data.costs, _pi, _sigma, _alpha, cut_duals, cut_indices, P)
        dec = eigen!(P)
        new_lambda_min = dec.values[1]
        if new_lambda_min - lambda_min > 1e-6
            lambda_min = new_lambda_min
        else
            break
        end
    end
    # @show P
    # @show eigvals(P)
    # write("P.bin", P)

    # return the safe bound
    if lambda_min < 0.0
        _pi += lambda_min
    end
    return data.fixed_cost + K * _pi + sum(_sigma) +
           (1 / (n - K + 1)) * sum(_alpha[i, i] for i in 1:n)
end

function solve(data::Data{Dim}, K::Int)::Solution where {Dim}
    println("Solving $(length(data.points)) points in $(Dim) dimensions")
    mat"maxNumCompThreads(1)"
    rng = MersenneTwister(12345678)

    # Set the constants
    n = length(data.points)

    # Build the MIP model
    model = Model(SDPSolver.Optimizer)
    if SDPSolverName == "SDPNAL"
        # set_optimizer_attribute(model, "printlevel", 0)
        # set_optimizer_attribute(model, "stopoption", 0)
    else
        set_optimizer_attribute(model, "verbose", false)
    end
    @variables(model, begin
        z[i = 1:n, j = 1:n] >= ((i == j) ? (1 / (n - K + 1)) : 0.0), PSD
    end)
    @objective(
        model,
        Min,
        -sum(data.costs[i, j] * z[i, j] for i in 1:n, j in 1:n)
    )
    @constraints(model, begin
        c_pi, sum(z[i, i] for i in 1:n) == K
        c_sigma[i = 1:n], sum(z[i, j] for j in 1:n) == 1
    end)

    # loop adding triangle cuts
    target_obj = Inf
    z_target = zeros(Float64, n, n)
    buffers = HeuristicBuffers{Dim}(n, K)
    alpha = 0.0 # 0.9
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
    cut_indices = Vector{Tuple{Int, Int, Int}}()
    _sigma = zeros(Float64, n)
    cut_duals = Float64[]
    cut_grad = Float64[]
    _alpha = zeros(Float64, n, n)
    P = zeros(Float64, n, n)
    cut_round = 0
    obj = 0.0
    curr_tol = init_tol
    min_viol = sqrt(curr_tol) * (K / n)
    tol_was_decreased = false
    best_bound = 0.0
    perm = collect(1:n)
    while true
        # solve the SDP relaxation
        if SDPSolverName == "SDPNAL"
            set_optimizer_attribute(model, "ADMtol", curr_tol / ph1_to_ph2_tol)
            set_optimizer_attribute(model, "tol", curr_tol)
        else
            set_optimizer_attribute(model, "eps_rel", curr_tol)
        end
        sdp_time = @elapsed optimize!(model)
        new_obj = data.fixed_cost + objective_value(model)
        nb_infeas = 0
        max_infeas = 0.0
        for i in 1:n, j in 1:n
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
        remain_cuts = length(added_cuts)
        for c in 1:remain_cuts
            # should_keep[c] = get_attribute(added_cuts[c], MOI.ConstraintBasisStatus()) == MOI.BASIC
            # should_keep[c] = abs(value(added_cuts[c])) < min_viol
            should_keep[c] = abs(dual(added_cuts[c])) > 1e-6 * new_obj
            # should_keep[c] = true
            # i, j, l = cut_indices[c]
            # if l == 0
            #     viol = z_[i, j] - z_[i, i]
            # else
            #     viol = z_[i, j] + z_[i, l] - z_[i, i] - z_[j, l]
            # end
            # should_keep[c] = viol > -min_viol
        end
        target_obj = run_rounding_heuristic(
            rng,
            data,
            z_,
            z_target,
            target_obj,
            buffers.centroids,
            buffers.points_to_cluster,
            buffers.cluster_sizes,
            buffers.unused,
        )

        # compute a safe bound and the gap
        _pi = dual(c_pi)
        for i in 1:n
            _sigma[i] = dual(c_sigma[i])
        end
        resize!(cut_duals, length(added_cuts))
        for c in 1:length(added_cuts)
            cut_duals[c] = max(0.0, dual(added_cuts[c]))
        end
        for i in 1:n, j in 1:n
            _alpha[i, j] = max(0.0, dual(LowerBoundRef(z[i, j])))
            if SDPSolverName == "SCS" && i != j
                _alpha[i, j] *= 0.5
            end
        end
        resize!(cut_grad, length(added_cuts))
        safe_bound = compute_safe_bound(data, _pi, _sigma, cut_duals, cut_indices, _alpha, K, P, cut_grad)
        best_bound = max(best_bound, safe_bound)
        gap = (target_obj - best_bound) / target_obj
        if gap <= gap_tol
            @show cut_round, target_obj, best_bound, curr_tol, sdp_time, gap
            break
        end

        # remove cuts that are not needed
        c = 1
        while c <= remain_cuts
            if !should_keep[c]
                JuMP.delete(model, added_cuts[c])
                added_cuts[c], added_cuts[end] = added_cuts[end], added_cuts[c]
                pop!(added_cuts)
                should_keep[c], should_keep[end] = should_keep[end], should_keep[c]
                pop!(should_keep)
                cut_indices[c], cut_indices[end] = cut_indices[end], cut_indices[c]
                pop!(cut_indices)
                remain_cuts -= 1
            end
            c += 1
        end

        # add new cuts
        cut_round += 1
        nb_cuts = 0
        first = true
        reduced_alpha = false
        last_cut = length(added_cuts)
        while first || reduced_alpha
            update_z_aux()
            nb_cuts = 0
            shuffle!(rng, perm)
            separate_pivot_cuts!(n, z_aux, pivot_cuts, min_viol, perm)
            resize!(pivot_cuts, min(target_nb_cuts, length(pivot_cuts)))
            for cut in pivot_cuts
                if zs_(cut.i, cut.i) >= zs_(cut.i, cut.j) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(added_cuts, @constraint(model, z[cut.i, cut.i] >= z[cut.i, cut.j]))
                push!(cut_indices, (cut.i, cut.j, 0))
                if nb_cuts >= target_nb_cuts
                    break
                end
            end
            shuffle!(rng, perm)
            separate_triangle_cuts!(n, z_aux, triangle_cuts, min_viol, perm)
            resize!(triangle_cuts, min(target_nb_cuts, length(triangle_cuts)))
            for cut in triangle_cuts
                if zs_(cut.j, cut.l) >= zs_(cut.i, cut.j) + zs_(cut.i, cut.l) - zs_(cut.i, cut.i) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(
                    added_cuts,
                    @constraint(model, z[cut.j, cut.l] >= z[cut.i, cut.j] + z[cut.i, cut.l] - z[cut.i, cut.i])
                )
                push!(cut_indices, (cut.i, cut.j, cut.l))
                if nb_cuts >= 2 * target_nb_cuts
                    break
                end
            end
            reduced_alpha = false
            if alpha != 0.0
                if nb_cuts < target_nb_cuts
                    alpha -= (1 - alpha) / 3
                    reduced_alpha = true
                    for c in (last_cut+1):length(added_cuts)
                        JuMP.delete(model, added_cuts[c])
                    end
                    resize!(added_cuts, last_cut)
                    resize!(cut_indices, last_cut)
                end
                if nb_cuts >= 2 * target_nb_cuts
                    alpha += (1 - alpha) / 5
                end
                if alpha < 0.1
                    alpha = 0.0
                end
            end
            first = false
        end
        resize!(should_keep, length(added_cuts))

        diff = round(safe_bound - obj, digits = 5)
        obj = safe_bound
        @show cut_round, target_obj, safe_bound, diff, nb_cuts, remain_cuts, alpha, curr_tol, sdp_time, gap
        if nb_cuts == 0 || (new_obj > target_obj) || (diff < 1e-6 * safe_bound && !tol_was_decreased)
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
