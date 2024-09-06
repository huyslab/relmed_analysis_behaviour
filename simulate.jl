function task_vars_for_condition(condition::String, split_by_confusion_time::Bool = false)
    # Load sequence from file
    task = DataFrame(CSV.File("data/PLT_task_structure_" * condition * ".csv"))

    # Renumber block
    task.block = task.block .+ (task.session .- 1) * maximum(task.block)

    # Arrange feedback by optimal / suboptimal
    task.feedback_optimal = 
        ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

    task.feedback_suboptimal = 
        ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)


    # Arrange outcomes such as second column is optimal
    outcomes = hcat(
        task.feedback_suboptimal,
        task.feedback_optimal,
    )

    if split_by_confusion_time        
        task.confusing_feedback = (task.optimal_A .== 1) .&& (task.feedback_A .< task.feedback_B) .|| (task.optimal_A .== 0) .&& (task.feedback_A .> task.feedback_B)
        # group by block, and then code the confusion time as the first trial where the feedback is confusing
        task[!, :confusion_time] .= missing

        # Group by block
        grouped_task = groupby(task, :block)

        # Iterate over each block
        for block in grouped_task
            # Find the first trial number with confusing feedback in the block
            first_confusing_trial = findfirst(block.confusing_feedback)
            if first_confusing_trial !== nothing
                # Update the confusion_time column for the block
                block.confusion_time .= block.trial[first_confusing_trial]
            end
        end
        task[!, :early_confusion] = task.confusion_time .<= 5
    end

    return (
        task = task,
        block = task.block,
        valence = unique(task[!, [:block, :valence]]).valence,
        outcomes = outcomes
    )

end

# Simulate data from model prior
function simulate_from_prior(
    n::Int64; # How many participants worth of data to simulate
    model::Function,
    block::Vector{Int64}, # Block number
    valence::AbstractVector, # Valence of each block
    outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
    initV::Matrix{Float64}, # Initial Q (and W) values
    set_size::Union{Vector{Int64}, Nothing} = nothing, # Set size for each block, required for WM models
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
    fixed_params::Dict = Dict(:ρ => nothing, :α => nothing),
    random_seed::Union{Int64, Nothing} = nothing
)

    # Trials per block
    n_trials = div(length(block), maximum(block))
    if length(block) % maximum(block) != 0
        n_trials += 1
    end

    # Prepare model for simulation
    prior_model = model(
        block = block,
        valence = valence,
        choice = fill(missing, length(block)),
        outcomes = outcomes,
        initV = initV,
        set_size = set_size,
        parameters = parameters,
        sigmas = sigmas,
        fixed_params = fixed_params
    )

    # Draw parameters and simulate choice
    prior_sample = sample(
        isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
        prior_model,
        Prior(),
        n
    )

    N = length(block)

    # pivot the columns in the prior_sample that start with "choice" longer
    # with another new column that is the trial number for each choice (i.e., from the column name)
    # and then stack them all together
    # choice_df = stack(prior_sample, [Symbol("choice[$i]") for i in 1:N], :trial_no, :choice)

    # Arrange choice for return
    sim_data = DataFrame(
        PID = repeat(1:n, inner = length(block)),
        block = repeat(block, n),
        valence = repeat(valence, inner = n_trials, outer = n),
        trial = repeat(1:n_trials, n * maximum(block))
    )

    for p in parameters
        if haskey(fixed_params, p)
            sim_data[!, p] = fill(fixed_params[p], n * N)
        else
            v = repeat(prior_sample[:, p, 1], inner = N)
            if haskey(transformed, p)
                v = v .|> a2α # assume all transformations are to [0, 1]
                sim_data[!, transformed[p]] = v
            else
                sim_data[!, p] = v
            end
        end
    end

    # Compute Q values
    gq = generated_quantities(prior_model, prior_sample)
    Qs = [pt.Qs for pt in gq] |> vec
    choices = [pt.choice for pt in gq] |> vec
    sim_data.choice = vcat([ch for ch in choices]...)

    sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...)
    sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...)

    if !isnothing(set_size)
        Ws = [pt.Ws for pt in gq] |> vec
        sim_data.W_optimal = vcat([ws[:, 2] for ws in Ws]...)
        sim_data.W_suboptimal = vcat([ws[:, 1] for ws in Ws]...)
    end

    return sim_data
end

function simulate_from_hierarchical_prior(
    n::Int64; # How many participants are therefore
    model::Function,
    block::Vector{Vector{Int64}}, # Block number
    valence::Vector{Vector{Float64}}, # Valence of each block
    outcomes::Vector{Matrix{Float64}}, # Outcomes for options, first column optimal
    initV::Matrix{Float64}, # Initial Q (and W) values
    set_size::Union{Vector{Vector{Int64}}, Nothing} = nothing, # Set size for each block, required for WM models
    parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
    random_seed::Union{Int64, Nothing} = nothing
)

    # Check lengths
    @assert length(block) == length(valence) "Number of participants not consistent"
    @assert length(block) == length(outcomes) "Number of participants not consistent"
    @assert all([length(b) for b in block] .== [size(o, 1) for o in outcomes]) "Number of trials not consistent"
    @assert all([maximum(block[s]) == length(valence[s]) for s in eachindex(block)]) "Number of blocks not consistent"

    # Trials per block
    n_trials = div(length(block[1]), maximum(block[1]))
    # if length(block[1]) % maximum(block[1]) != 0
    #     n_trials += 1
    # end

    prior_model = model(
        block = block,
        valence = valence,
        choice = [fill(missing, length(block[s])) for s in eachindex(block)],
        outcomes = outcomes,
        initV = initV,
        set_size = set_size,
        parameters = parameters,
        sigmas = sigmas
    )

    # Draw parameters and simulate choice
    prior_sample = sample(
        isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
        prior_model,
        Prior(),
        1
    )

    cm = [countmap(block[s]) for s in eachindex(block)]
    @assert allequal(cm) "Number of trials per block not consistent"

    # Arrange choice for return
    sim_data = DataFrame(
        PID = repeat(1:n, inner = length(block[1])),
        block = vcat(block...),
        valence = vcat([vcat([fill(valence[s][i], cm[s][i]) for i in eachindex(valence[s])]...) for s in 1:n]...),
        trial = vcat([repeat(1:n_trials, maximum(block[s])) for s in 1:n]...),
        choice = vec(Array(prior_sample[:, [Symbol("choice[$s][$i]") for s in 1:n for i in eachindex(block[s])], 1]))
    )

    for p in parameters
        v = vcat([repeat(Array(prior_sample[:, [Symbol("$p[$s]")], 1]), inner = sum(length(b))) for s in 1:n for b in eachindex(block[s])]...)
        if haskey(transformed, p)
            v = v .|> a2α # assume all transformations are to [0, 1]
            sim_data[!, transformed[p]] = v
        else
            sim_data[!, p] = v
        end
    end

    # Compute Q values
    gq = generated_quantities(prior_model, prior_sample)[1, 1]
    Qs = vcat([gq.Qs[i] for i in eachindex(gq.Qs)]...)

    sim_data.Q_optimal, sim_data.Q_suboptimal = Qs[:, 2], Qs[:, 1]

    if !isnothing(set_size)
        Ws = vcat([gq.Ws[i] for i in eachindex(gq.Ws)]...)
        sim_data.W_optimal, sim_data.W_suboptimal = Ws[:, 2], Ws[:, 1]
    end

    return sim_data

end

# Prepare pilot data for fititng with model
function prepare_for_fit(data)

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

	return forfit, pids
end

# # Sample from posterior conditioned on DataFrame with data for single participant
function posterior_sample_single_p(
	data::AbstractDataFrame;
    model::Function,
	initV::Float64,
    set_size::Union{Vector{Int64}, Nothing} = nothing,
    parameters::Vector{Symbol} = [:ρ, :a],
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 2., :a => 0.5),
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 1000
)
    posterior_model = model(;
        block = data.block,
        valence = unique(data[!, [:block, :valence]]).valence,
        choice = data.choice,
        outcomes = hcat(
            data.feedback_suboptimal,
            data.feedback_optimal,
        ),
        initV = fill(initV, 1, 2),
        set_size = set_size,
        parameters = parameters,
        sigmas = sigmas
    )

	fit = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		posterior_model, 
		NUTS(), 
		MCMCThreads(), 
		iter_sampling, 
		4)

	return fit
end

# Find MLE / MAP for DataFrame with data for single participant
function optimize_single_p_QL(
	data::AbstractDataFrame;
	initV::Float64,
	estimate::String = "MAP",
    model::Function = RL_ss,
    set_size::Union{Vector{Int64}, Nothing} = nothing,
	initial_params::Union{AbstractVector,Nothing} = nothing,
	parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5)
)
	res = model(;
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2),
        set_size = set_size,
		parameters = parameters,
        sigmas = sigmas
    )

	if estimate == "MLE"
		fit = maximum_likelihood(res; initial_params = initial_params)
	elseif estimate == "MAP"
		fit = maximum_a_posteriori(res; initial_params = initial_params)
	end

	return fit
end

# Find MLE / MAP multiple times
function optimize_multiple_single_p_QL(
	data::DataFrame;
	initV::Float64,
	estimate::String = "MAP",
    model::Function = RL_ss,
    set_size::Union{Vector{Int64}, Nothing} = nothing,
	initial_params::Union{AbstractVector,Nothing}=[mean(truncated(Normal(0., 2.), lower = 0.)), 0.5],
	include_true::Bool = true, # Whether to return true value if this is simulation
	parameters::Vector{Symbol} = [:ρ, :a], # Group-level parameters to estimate
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    sigmas::Dict{Symbol, Float64} = Dict(:ρ => 1., :a => 0.5)
)
	ests = []
	lk = ReentrantLock()

	Threads.@threads for p in unique(data.PID)

		# Select data
		gdf = filter(x -> x.PID == p, data)

		# Optimize
		est = optimize_single_p_QL(
			gdf;
			initV = initV,
			estimate = estimate,
            model = model,
			initial_params = initial_params,
            set_size = set_size,
            parameters = parameters,
            sigmas = sigmas
		)

		# Return
        est_dct = Dict{Symbol, Union{Int64, Float64}}(:PID => gdf.PID[1])
        if include_true
            for p in parameters
                est_dct[Symbol("true_$p")] = haskey(transformed, p) ? α2a(gdf[!, transformed[p]][1]) : gdf[!, p][1]
                est_dct[Symbol("MLE_$p")] = est.values[p]
            end
        else
            for p in parameters
                est_dct[p] = est.values[p]
            end
        end

		lock(lk) do
            est = NamedTuple{Tuple(keys(est_dct))}(values(est_dct))
			push!(ests, est)
		end
	end

	return DataFrame(ests)
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
