# Simulating data and task structures for PILT tasks -------------------------------------------------------------------
function task_vars_for_condition(condition::String)
	# Load sequence from file
	task = DataFrame(CSV.File("data/PLT_task_structure_$condition.csv"))

	# Renumber block
	task.block = task.block .+ (task.session .- 1) * maximum(task.block)

	# Arrange feedback by optimal / suboptimal
	task.feedback_optimal = 
		ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

	task.feedback_suboptimal = 
		ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)

    # currently assume one pair per block
    task.pair .= 1

	# Arrange outcomes such as second column is optimal
	# outcomes = hcat(task.feedback_suboptimal, task.feedback_optimal)

	return task

end

function random_sequence(;
	optimal::Vector{Float64},
	suboptimal::Vector{Float64},
	n_confusing::Int64,
	n_trials::Int64,
    n_pairs::Int64 = 1
)
    seq = Vector{Matrix{Float64}}(undef, n_pairs)
    for i in 1:n_pairs
        common = shuffle(
            vcat(
                fill(true, n_trials - n_confusing),
                fill(false, n_confusing)
            )
        )
        opt_seq = ifelse.(
            common, 
            shuffle(collect(Iterators.take(Iterators.cycle(optimal), n_trials))), 
            shuffle(collect(Iterators.take(Iterators.cycle(suboptimal), n_trials)))
        )
        subopt_seq = ifelse.(
            common, 
            shuffle(collect(Iterators.take(Iterators.cycle(suboptimal), n_trials))),
            shuffle(collect(Iterators.take(Iterators.cycle(optimal), n_trials))),
        )
        seq[i] = hcat(subopt_seq, opt_seq, fill(i, n_trials))
    end
    seq = DataFrame(vcat(seq...)[shuffle(1:end), :], :auto)
	return DataFrames.transform(groupby(seq, :x3), eachindex => :x4) |> Matrix
end

function random_outcomes_from_sequence(;
    n_blocks::Int64,
    set_sizes::Vector{Int64},
    kwargs...
)

return vcat(
    [
        random_sequence(;
            optimal = mgnt[2], 
            suboptimal = mgnt[1],
            n_pairs = set_sizes[i] ÷ 2,
            kwargs...
        )
        for (i, mgnt) in enumerate(
            Iterators.take(
                Iterators.cycle(
                    [
                        ([0.01], [0.5, 1.]),
                        ([0.01, 0.5], [1.]),
                        ([-0.5, -1.], [-0.01]),
                        ([-1.], [-0.01, -0.5])
                    ]
                ), n_blocks
            )
        )
    ]...
)

end

function set_size_per_block(;
    set_sizes::Union{Int64, Vector{Int64}},
    n_blocks::Int64
)
    @assert all([s%2 == 0 for s in set_sizes]) "Set sizes must be even"
    if length(set_sizes) == n_blocks
        ssz = set_sizes
    elseif set_sizes isa Int64
        ssz = fill(set_sizes, n_blocks)
    else
        ssz = sort(
            vcat(
                set_sizes,
                sample(
                    set_sizes,
                    aweights(fill(1, length(set_sizes))),
                    n_blocks - length(set_sizes),
                    replace = true
                )
            )
        )
    end
    return ssz
end

function create_random_task(;
    n_confusing::Int64,
	n_trials::Int64,
	n_blocks::Int64,
    set_sizes::Union{Int64, Vector{Int64}} = 2
)

    set_sizes_blk = set_size_per_block(set_sizes = set_sizes, n_blocks = n_blocks)
	# Create task sequence
	block = vcat([i for i in 1:n_blocks for _ in 1:(n_trials * (set_sizes_blk[i] ÷ 2))]...)
	outcomes = random_outcomes_from_sequence(
        n_confusing = n_confusing,
        n_trials = n_trials,
        n_blocks = n_blocks,
        set_sizes = set_sizes_blk
    )
	valence = sign.(outcomes[[findfirst(block .== i) for i in 1:n_blocks], 1])
	trial = Int.(outcomes[:, 4])

	df = DataFrame(
		block = block,
		trial = trial,
        pair = Int.(outcomes[:, 3]),
		valence = valence[block],
		feedback_optimal = outcomes[:, 2],
		feedback_suboptimal = outcomes[:, 1],
        set_size = set_sizes_blk[block]
	)

	return df
end

function simulate_from_prior(
    N::Int64; # number of simulated participants or repeats (depending on priors)
    model::Function,
    priors::Dict,
    parameters::Vector{Symbol} = collect(keys(priors)),
    initial::Union{Float64, Nothing} = nothing,
    transformed::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}(), # Transformed parameters
    condition::Union{String, Nothing} = nothing,
    fixed_struct::Union{DataFrame, Nothing} = nothing,
    structure::NamedTuple = (
        n_blocks = 48, n_trials = 13, n_confusing = 3, set_sizes = 2
    ),
    repeats::Bool = false,
    gq::Bool = false,
    random_seed::Union{Int64, Nothing} = nothing
)	
    # Load sequence from file or generate random task
    if !isnothing(condition)
        task_strct = task_vars_for_condition(condition)
        task_strct.set_size = set_size_per_block(
            set_sizes = structure.set_sizes,
            n_blocks = maximum(task_strct.block)
        )[task_strct.block]
    elseif !isnothing(fixed_struct)
        task_strct = fixed_struct
    else
        task_strct = create_random_task(;
            structure...
        )
    end

    # define model
    chce = fill(missing, nrow(task_strct))
    data_to_fit = unpack_data(task_strct)
    prior_model = model(
        data_to_fit,
        chce;
        priors = priors,
        initial = initial
    )

    # Draw parameters and simulate choice
    prior_sample = sample(
        isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
        prior_model,
        Prior(),
        N
    )

    nT = length(data_to_fit.block)

    # Arrange choice for return
    sim_data = DataFrame(
        PID = repeat(1:N, inner = nT),
        block = repeat(task_strct.block, N),
        valence = repeat(task_strct.valence, N),
        trial = repeat(task_strct.trial, N),
        pair = repeat(task_strct.pair, N),
        set_size = repeat(task_strct.set_size, N),
    )

    for p in parameters
        v = repeat(prior_sample[:, p, 1], inner = nT)
        if haskey(transformed, p)
            v = v .|> a2α # assume all transformations are to [0, 1]
            sim_data[!, transformed[p]] = v
        else
            sim_data[!, p] = v
        end
    end

    sim_data = leftjoin(sim_data, 
        task_strct[!, [:block, :trial, :pair, :feedback_optimal, :feedback_suboptimal]],
        on = [:block, :trial, :pair]
    )

    # Renumber blocks
    if repeats && N > 1
        sim_data.block = sim_data.block .+ 
            (sim_data.PID .- 1) .* maximum(sim_data.block)
    end

    if gq
        # Compute Q values
        gq = generated_quantities(prior_model, prior_sample)
        Qs = [pt.Qs for pt in gq] |> vec
        choices = [pt.choice for pt in gq] |> vec
        sim_data.choice = vcat([ch for ch in choices]...)

        # loglik = [pt.loglike for pt in gq] |> vec

        if !any([isnothing(q) for q in Qs])
            sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...)
            sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...)
        end

        if haskey(priors, :C)
            Ws = [pt.Ws for pt in gq] |> vec
            sim_data.W_optimal = vcat([ws[:, 2] for ws in Ws]...)
            sim_data.W_suboptimal = vcat([ws[:, 1] for ws in Ws]...)
        end
    end
    return sim_data
end


# will require work.
# function simulate_from_hierarchical_prior(
#     n::Int64; # How many participants are therefore
#     model::Function,
#     block::Vector{Vector{Int64}}, # Block number
#     valence::Vector{Vector{Float64}}, # Valence of each block
#     outcomes::Vector{Matrix{Float64}}, # Outcomes for options, first column optimal
#     initV::Matrix{Float64}, # Initial Q (and W) values
#     set_size::Union{Vector{Vector{Int64}}, Nothing} = nothing, # Set size for each block, required for WM models
#     parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
#     transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
#     sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
#     random_seed::Union{Int64, Nothing} = nothing
# )

#     # Check lengths
#     @assert length(block) == length(valence) "Number of participants not consistent"
#     @assert length(block) == length(outcomes) "Number of participants not consistent"
#     @assert all([length(b) for b in block] .== [size(o, 1) for o in outcomes]) "Number of trials not consistent"
#     @assert all([maximum(block[s]) == length(valence[s]) for s in eachindex(block)]) "Number of blocks not consistent"

#     # Trials per block
#     n_trials = div(length(block[1]), maximum(block[1]))
#     if length(block[1]) % maximum(block[1]) != 0
#         n_trials += 1
#     end

#     prior_model = model(
#         block = block,
#         valence = valence,
#         choice = [fill(missing, length(block[s])) for s in eachindex(block)],
#         outcomes = outcomes,
#         initV = initV,
#         set_size = set_size,
#         parameters = parameters,
#         sigmas = sigmas
#     )

#     # Draw parameters and simulate choice
#     prior_sample = sample(
#         isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
#         prior_model,
#         Prior(),
#         1
#     )

#     cm = [countmap(block[s]) for s in eachindex(block)]
#     @assert allequal(cm) "Number of trials per block not consistent"

#     # Arrange choice for return
#     sim_data = DataFrame(
#         PID = repeat(1:n, inner = length(block[1])),
#         block = vcat(block...),
#         valence = vcat([vcat([fill(valence[s][i], cm[s][i]) for i in eachindex(valence[s])]...) for s in 1:n]...),
#         trial = vcat([repeat(1:n_trials, maximum(block[s])) for s in 1:n]...),
#         choice = vec(Array(prior_sample[:, [Symbol("choice[$s][$i]") for s in 1:n for i in eachindex(block[s])], 1]))
#     )

#     for p in parameters
#         v = vcat([repeat(Array(prior_sample[:, [Symbol("$p[$s]")], 1]), inner = sum(length(b))) for s in 1:n for b in eachindex(block[s])]...)
#         if haskey(transformed, p)
#             v = v .|> a2α # assume all transformations are to [0, 1]
#             sim_data[!, transformed[p]] = v
#         else
#             sim_data[!, p] = v
#         end
#     end

#     # Compute Q values
#     gq = generated_quantities(prior_model, prior_sample)[1, 1]
#     Qs = vcat([gq.Qs[i] for i in eachindex(gq.Qs)]...)

#     sim_data.Q_optimal, sim_data.Q_suboptimal = Qs[:, 2], Qs[:, 1]

#     if !isnothing(set_size)
#         Ws = vcat([gq.Ws[i] for i in eachindex(gq.Ws)]...)
#         sim_data.W_optimal, sim_data.W_suboptimal = Ws[:, 2], Ws[:, 1]
#     end

#     return sim_data

# end

# Prepare pilot data for fititng with model
function prepare_for_fit(data; pilot2::Bool = false)

    data.condition .= pilot2 ? 1 : data.condition
	forfit = select(data, [:prolific_pid, :condition, :session, :block, :valence, :trial, :optimalRight, :outcomeLeft, :outcomeRight, :isOptimal])

	rename!(forfit, :isOptimal => :choice)

	# Make sure block is numbered correctly
	renumber_block(x) = indexin(x, sort(unique(x)))
	DataFrames.transform!(
		groupby(forfit, [:prolific_pid, :session]),
		:block => renumber_block => :block
	)

	# Arrange feedback by optimal / suboptimal
	forfit.feedback_optimal = 
		ifelse.(forfit.optimalRight .== 1, forfit.outcomeRight, forfit.outcomeLeft)

	forfit.feedback_suboptimal = 
		ifelse.(forfit.optimalRight .== 0, forfit.outcomeRight, forfit.outcomeLeft)

	# PID as number
	pids = unique(forfit[!, [:prolific_pid, :condition]])

	pids.PID = 1:nrow(pids)

	forfit = innerjoin(forfit, pids[!, [:prolific_pid, :PID]], on = :prolific_pid)

	# Block as Int64
	forfit.block = convert(Vector{Int64}, forfit.block)

    if pilot2
        forfit.set_size = data.n_pairs .* 2
	    forfit.pair = data.stimulus_pair
    end

	return forfit, pids
end

# # Sample from posterior conditioned on DataFrame with data for single participant
# function posterior_sample_single_p(
# 	data::AbstractDataFrame;
#     model::Function,
# 	initV::Float64,
#     set_size::Union{Vector{Int64}, Nothing} = nothing,
#     parameters::Vector{Symbol} = [:ρ, :a],
#     sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
# 	random_seed::Union{Int64, Nothing} = nothing,
# 	iter_sampling = 1000
# )
#     posterior_model = model(;
#         block = data.block,
#         valence = unique(data[!, [:block, :valence]]).valence,
#         choice = data.choice,
#         outcomes = hcat(
#             data.feedback_suboptimal,
#             data.feedback_optimal,
#         ),
#         set_size = set_sizes,
#         initV = fill(initial, 1, maximum(set_sizes)),
#         parameters = collect(keys(priors)),
#         priors = priors
#     )

# 	fit = sample(
# 		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
# 		posterior_model, 
# 		NUTS(), 
# 		MCMCThreads(), 
# 		iter_sampling, 
# 		4)

# 	return fit
# end

# Find MLE / MAP for DataFrame with data for single participant
function optimize_ss(
	data::AbstractDataFrame;
    initial::Union{Float64, Nothing} = nothing,
    model::Function = RL_ss,
	estimate::String = "MAP",
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    parameters::Vector{Symbol} = collect(keys(priors)),
    transformed::Dict{Symbol, Symbol} = Dict{:a => :α}, # Transformed parameters
    bootstraps::Int64 = 0
)
    data_for_fit = unpack_data(data)
    res = model(
        data_for_fit,
        data.choice;
        priors = priors,
        initial = initial
    )

    if estimate == "MLE"
        fit = maximum_likelihood(res)
    elseif estimate == "MAP"
        fit = maximum_a_posteriori(res)
    end
    
    # Estimate covariance matrix using bootstrapping?
    cov_mat = Matrix{Float64}(undef, length(parameters), length(parameters))
    if bootstraps > 0
        ests_tr = Matrix{Float64}(undef, bootstraps, length(parameters))
        Threads.@threads for b in 1:bootstraps
            # sample rows with replacement
            idxs = sample(Xoshiro(b), 1:nrow(data), nrow(data), replace=true)
            data_boot = data[idxs, :]

            boot_res = model(
                data_for_fit,
                data_boot.choice; # sample with replacement from choices
                priors = priors,
                initial = initial
            )

            if estimate == "MLE"
                boot_fit = maximum_likelihood(boot_res)
            elseif estimate == "MAP"
                boot_fit = maximum_a_posteriori(boot_res)
            end

            ests_tr[b, :] = [haskey(transformed, p) ? a2α(boot_fit.values[p]) : boot_fit.values[p] for p in parameters]
        end
        cov_mat = cov(ests_tr, corrected = false) # covariance of transformed parameters
    end

    # get predictions from the model for choices by setting a Dirac prior on the parameters
    priors_fitted = Dict{Symbol, Distribution}()
    for p in parameters
        priors_fitted = merge!(priors_fitted, Dict(p => Dirac(fit.values[p])))
    end
    
    gq_mod = model(
        data_for_fit,
        fill(missing, length(data.block));
        priors = priors_fitted,
        initial = initial
    )

    # Draw parameters and simulate choice
    gq_samp = sample(
        Random.default_rng(),
        gq_mod,
        Prior(),
        1
    )
    gq = generated_quantities(gq_mod, gq_samp)
    choices = gq[1].choice
    loglike = gq[1].loglike
    bic = -2 * loglike + length(parameters) * log(nrow(data))

	return fit, bic, choices, loglike, cov_mat
end

# Find MLE / MAP multiple times
function optimize_multiple(
	data::AbstractDataFrame;
    initial::Union{Float64, Nothing} = nothing,
    model::Function = RL_ss,
    estimate::String = "MAP",
	include_true::Bool = true, # Whether to return true value if this is simulation
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    parameters::Vector{Symbol} = collect(keys(priors)),
    bootstraps::Int64 = 0
)
	ests = []
    choice_df = Dict{Int64, DataFrame}()
    cov_mat_dict = Dict{Int64, Matrix{Float64}}()
	lk = ReentrantLock()

	Threads.@threads for p in unique(data.PID)

		# Select data
		gdf = filter(x -> x.PID == p, data)

		# Optimize
		est, bic, choices, loglike, cov_mat = optimize_ss(
			gdf;
			initial = initial,
            model = model,
			estimate = estimate,
            priors = priors,
            parameters = parameters,
            transformed = transformed,
            bootstraps = bootstraps
		)

		# Return
        est_dct = Dict{Symbol, Union{Int64, Float64}}(:PID => gdf.PID[1])
        est_par = Dict{Symbol, Symbol}()
        if include_true
            for p in parameters
                est_par[p] = haskey(transformed, p) ? transformed[p] : p
                est_dct[Symbol("true_$(est_par[p])")] = gdf[!, est_par[p]][1]
                est_dct[Symbol("MLE_$(est_par[p])")] = haskey(transformed, p) ? a2α(est.values[p]) : est.values[p]
            end
        else
            for p in parameters
                est_par[p] = haskey(transformed, p) ? transformed[p] : p
                est_dct[est_par[p]] = haskey(transformed, p) ? a2α(est.values[p]) : est.values[p]
            end
        end

        est_dct[:loglike] = loglike
        est_dct[:BIC] = bic

        if bootstraps > 0
            # add bootstrap standard deviations to output dictionary
            for (i, p) in enumerate(parameters)
                est_dct[Symbol("sd_$(est_par[p])")] = sqrt(cov_mat[i, i])
            end
        end
		lock(lk) do
            est = NamedTuple{Tuple(keys(est_dct))}(values(est_dct))
			push!(ests, est)
            merge!(cov_mat_dict, Dict(p => cov_mat))
            choice_df[p] = DataFrame(
                PID = gdf.PID,
                block = gdf.block,
                valence = gdf.valence,
                trial = gdf.trial,
                true_choice = gdf.choice,
                predicted_choice = choices
            )
		end
	end

    # make choice_df into a single DataFrame
    choice_df = vcat(values(choice_df)...)

	return DataFrame(ests), choice_df, cov_mat_dict
end

# function bootstrap_optimize_single_p_QL(
# 	PLT_data::DataFrame;
# 	n_bootstrap::Int64 = 20,
# 	estimate = "MAP",
# 	prior_ρ::Distribution,
# 	prior_a::Distribution
# 	)
	
# 		# Initial value for Q values
# 	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# 	prolific_pids = sort(unique(PLT_data.prolific_pid))

# 	bootstraps = []

# 	for i in 1:n_bootstrap
		
# 		# Resample the data with replacement
# 		idxs = sample(Xoshiro(i), prolific_pids, length(prolific_pids), replace=true)

# 		tdata = filter(x -> x.prolific_pid in prolific_pids, PLT_data)

# 		forfit, pids = prepare_for_fit(PLT_data)

# 		# Randomly sample initial parameters
# 		initial_params = [rand(truncated(Normal(0., 5.), lower = 0.)), rand(Normal(0., 1.))]
		
# 		tfit = optimize_multiple_single_p_QL(
# 				forfit;
# 				initV = aao,
# 				estimate = estimate,
# 				initial_params = initial_params,
#                 set_size = nothing,
#                 include_true = true

# 			)

# 		tfit = innerjoin(tfit, pids, on = :PID)

# 		tfit[!, :bootstrap_idx] .= i
		
# 		# Compute the correlation for the resampled data
# 		push!(bootstraps, tfit)
# 	end

# 	return vcat(bootstraps...)

# end