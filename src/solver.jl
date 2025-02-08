struct Solution
    cost::Float64
    clusters::Vector{Vector{Int}}
end

const min_viol = 1e-3

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

    # Set the constants
    n = length(data.points)

    # Build the MIP model
    model = Model(SDPNAL.Optimizer)
    # set_optimizer_attribute(model, "warm_start", true)
    @variables(model, begin
        z[1:n, 1:n] >= 0, PSD
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
        c6[i = 1:n], z[i, i] >= (K / (n - K + 1))
        # c7[i = 1:n, j = 1:n; i != j], z[i, j] >= 0
    end)

    # loop adding triangle cuts
    triangle_cuts = Vector{TriangleCut}()
    all_triangle_cuts = Vector{TriangleCut}()
    all_triangle_refs = Vector{ConstraintRef}()
    z_ = zeros(n, n)
    while true
        optimize!(model)
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
        # if nb_infeas == 0
        if !isempty(triangle_cuts)
            # if true
            break
        end
        @show nb_infeas
        @show max_infeas
        break

        separate_triangle_cuts!(n, z_, triangle_cuts)
        resize!(triangle_cuts, min(100 * n, length(triangle_cuts)))
        if length(triangle_cuts) < n
            break
        end
        for cut in triangle_cuts
            c_ref = @constraint(model, z[cut.j, cut.l] >= z[cut.i, cut.j] + z[cut.i, cut.l] - z[cut.i, cut.i])
            push!(all_triangle_cuts, cut)
            push!(all_triangle_refs, c_ref)
        end
    end
    for i in 1:n, j in 1:n
        z_[i, j] = value(z[i, j])
    end
    @show data.fixed_cost + objective_value(model)
    return compute_and_check_solution(data, z_, data.fixed_cost + objective_value(model))

    # identify the active cuts
    active_pivot_cuts = Vector{PivotCut}()
    # for i in 1:n, j in 1:n
    #     if z_[i, j] > z_[i, i] - 1e-3
    #         push!(active_pivot_cuts, PivotCut(i, j, z_[i, j] - z_[i, i]))
    #     end
    # end
    @show length(active_pivot_cuts)
    active_triangle_cuts = Vector{TriangleCut}()
    # for cut in all_triangle_cuts
    #     if z_[cut.j, cut.l] < z_[cut.i, cut.j] + z_[cut.i, cut.l] - z_[cut.i, cut.i] + 1e-3
    #         push!(active_triangle_cuts, cut)
    #     end
    # end
    @show length(active_triangle_cuts)
    active_sdp_cuts = Vector{SdpCut}()
    # eigen_vals_and_vecs = eigen(z_)
    # for i in 1:n
    #     vec = eigen_vals_and_vecs.vectors[:, i]
    #     if eigen_vals_and_vecs.values[i] < 1e-3
    #         push!(active_sdp_cuts, SdpCut(vec))
    #     end
    # end
    @show length(active_sdp_cuts)

    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 0)
    @variables(model, begin
        z[i = 1:n, j = i:n] >= 0
        0 <= y[i = 1:n, j = (i+1):n] <= 1
    end)
    @objective(
        model,
        Min,
        data.fixed_cost - (K / n) * sum(data.costs[i, i] * z[i, i] for i in 1:n) -
        (2 * K / n) * sum(data.costs[i, j] * z[i, j] for i in 1:n, j in (i+1):n)
    )
    apc = active_pivot_cuts
    atc = active_triangle_cuts
    asc = active_sdp_cuts
    zs(i::Int, j::Int) = (j < i) ? z[j, i] : z[i, j]
    @constraints(
        model,
        begin
            c1, sum(z[i, i] for i in 1:n) == n
            c2[i = 1:n], sum(z[j, i] for j in 1:(i-1)) + sum(z[i, j] for j in i:n) == (n / K)
            c3[l = 1:length(apc)], zs(apc[l].i, apc[l].i) >= zs(apc[l].i, apc[l].j)
            c6[i = 1:n], z[i, i] >= (K / (n - K + 1))
            c7[l = 1:length(atc)],
            zs(atc[l].j, atc[l].l) >= zs(atc[l].i, atc[l].j) + zs(atc[l].i, atc[l].l) - zs(atc[l].i, atc[l].i)
            c8[l = 1:length(asc)],
            sum(asc[l].vec[i]^2 * z[i, i] for i in 1:n) +
            sum(2 * asc[l].vec[i] * asc[l].vec[j] * z[i, j] for i in 1:n, j in (i+1):n) >= 0
        end
    )

    # loop adding triangle cuts
    triangle_cuts = Vector{TriangleCut}()
    all_triangle_cuts = Vector{TriangleCut}()
    z_target = z_
    alpha = 0.99
    z_ = zeros(n, n)
    z_aux = zeros(n, n)
    pivot_cuts = Vector{PivotCut}()
    zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    function update_z_aux()
        for i in 1:n, j in 1:n
            z_aux[i, j] = alpha * z_target[i, j] + (1 - alpha) * z_[i, j]
        end
        return nothing
    end
    added_cuts = Vector{ConstraintRef}()
    is_basic = Vector{Bool}()
    while true
        optimize!(model)
        obj = objective_value(model)
        remain_cuts = length(added_cuts)
        for c in 1:remain_cuts
            is_basic[c] = get_attribute(added_cuts[c], MOI.ConstraintBasisStatus()) == MOI.BASIC
        end
        c = 1
        while c <= remain_cuts
            if is_basic[c]
                delete(model, added_cuts[c])
                added_cuts[c], added_cuts[end] = added_cuts[end], added_cuts[c]
                pop!(added_cuts)
                is_basic[c], is_basic[end] = is_basic[end], is_basic[c]
                pop!(is_basic)
                remain_cuts -= 1
            end
            c += 1
        end
        optimize!(model)
        if abs(obj - objective_value(model)) > 1e-6 * max(obj, objective_value(model))
            println("ERROR: objective value changed from $obj to $(objective_value(model))")
        end
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

        nb_cuts = 0
        while alpha > 0.0 && nb_cuts == 0
            update_z_aux()
            nb_cuts = 0
            separate_pivot_cuts!(n, z_aux, pivot_cuts)
            resize!(pivot_cuts, min(8 * n, length(pivot_cuts)))
            for cut in pivot_cuts
                if zs_(cut.i, cut.i) >= zs_(cut.i, cut.j) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(added_cuts, @constraint(model, zs(cut.i, cut.i) >= zs(cut.i, cut.j)))
                if nb_cuts >= n
                    break
                end
            end
            separate_triangle_cuts!(n, z_aux, triangle_cuts)
            resize!(triangle_cuts, min(8 * n, length(triangle_cuts)))
            for cut in triangle_cuts
                if zs_(cut.j, cut.l) >= zs_(cut.i, cut.j) + zs_(cut.i, cut.l) - zs_(cut.i, cut.i) - min_viol
                    continue
                end
                nb_cuts += 1
                push!(
                    added_cuts,
                    @constraint(model, zs(cut.j, cut.l) >= zs(cut.i, cut.j) + zs(cut.i, cut.l) - zs(cut.i, cut.i))
                )
                if nb_cuts > 2 * n
                    break
                end
            end
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
        resize!(is_basic, length(added_cuts))

        @show obj, nb_cuts, remain_cuts, alpha
        if nb_cuts == 0
            break
        end
    end
    @constraints(
        model,
        begin
            c4[i = 1:n, j = (i+1):n], z[i, j] <= ((n - K + 1) / K) * y[i, j]
            c5_1[i = 1:n, j = (i+1):n], z[i, i] - z[i, j] <= ((n - K + 1) / K) * (1 - y[i, j])
            c5_2[i = 1:n, j = (i+1):n], z[j, j] - z[i, j] <= ((n - K + 1) / K) * (1 - y[i, j])
        end
    )
    for i in 1:n
        for j in (i+1):n
            set_integer(y[i, j])
        end
    end
    set_optimizer_attribute(model, "CPX_PARAM_SCRIND", 1)
    set_optimizer_attribute(model, "CPX_PARAM_EACHCUTLIM", 0)
    set_optimizer_attribute(model, "CPXPARAM_MIP_Strategy_HeuristicEffort", 0.2)
    set_optimizer_attribute(model, "CPX_PARAM_STARTALG", 0)
    set_optimizer_attribute(model, "CPX_PARAM_MIPDISPLAY", 4)

    # Set the cut separation callback
    # z_target = z_
    # alpha = 0.995
    # z_ = zeros(n, n)
    # z_aux = zeros(n, n)
    # pivot_cuts = Vector{PivotCut}()
    # zs_(i::Int, j::Int) = (j < i) ? z_[j, i] : z_[i, j]
    should_cut = true
    function cut_separation_callback(cb_data, is_cut::Bool)
        if !should_cut
            return
        end
        for i in 1:n, j in i:n
            z_[i, j] = callback_value(cb_data, z[i, j])
        end
        nb_cuts = 0
        while true
            update_z_aux()
            nb_cuts = 0
            separate_pivot_cuts!(n, z_aux, pivot_cuts)
            resize!(pivot_cuts, min(1000, length(pivot_cuts)))
            for cut in pivot_cuts
                if zs_(cut.i, cut.i) >= zs_(cut.i, cut.j) - min_viol
                    continue
                end
                nb_cuts += 1
                ctr_type = is_cut ? MOI.UserCut(cb_data) : MOI.LazyConstraint(cb_data)
                MOI.submit(model, ctr_type, @build_constraint(zs(cut.i, cut.i) >= zs(cut.i, cut.j)))
            end
            separate_triangle_cuts!(n, z_aux, triangle_cuts)
            resize!(triangle_cuts, min(1000, length(triangle_cuts)))
            for cut in triangle_cuts
                if zs_(cut.j, cut.l) >= zs_(cut.i, cut.j) + zs_(cut.i, cut.l) - zs_(cut.i, cut.i) - min_viol
                    continue
                end
                nb_cuts += 1
                ctr_type = is_cut ? MOI.UserCut(cb_data) : MOI.LazyConstraint(cb_data)
                lhs = zs(cut.j, cut.l)
                rhs = zs(cut.i, cut.j) + zs(cut.i, cut.l) - zs(cut.i, cut.i)
                MOI.submit(model, ctr_type, @build_constraint(lhs >= rhs))
            end
            if nb_cuts < 100
                if alpha < 1e-5
                    break
                end
                alpha -= 1 - alpha
                if alpha < 0.95
                    alpha = 0.0
                end
                println("Reducing alpha to $alpha")
            else
                break
            end
        end
        if nb_cuts == 0
            should_cut = false
        end
    end
    set_attribute(
        model,
        MOI.UserCutCallback(),
        (cb_data,) -> cut_separation_callback(cb_data, true),
    )
    set_attribute(
        model,
        MOI.LazyConstraintCallback(),
        (cb_data,) -> cut_separation_callback(cb_data, false),
    )

    # optimize and return the result
    optimize!(model)
    # println("z_ = $z_")
    # @show objective_value(model)
    # @show -(K / n) * 1e-3 * sum(data.costs[i, j] * z_[i, j] for i in 1:n, j in 1:n)
    return compute_and_check_solution(data, z_, objective_value(model))
end
