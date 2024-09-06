# This file contains a Q learning model for single participant data, 
# and methods for sampling, fitting, and computing quanitities based on it.

# Simulate data from model prior
function simulate_single_p_QL(
	n::Int64; # How many datasets to simulate
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing,
	prior_ρ::Distribution,
	prior_a::Distribution
)
	return sim_data = simulate_single_p_PILT(
		n;
		model = single_p_QL,
		block = block,
		valence = valence,
		outcomes = outcomes,
		initV = initV,
		random_seed = random_seed,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)
			
end

# Sample from posterior conditioned on DataFrame with data for single participant
function posterior_sample_single_p_QL(
	data::AbstractDataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 1000,
	prior_ρ::Distribution = truncated(Normal(0., 2.), lower = 0.),
	prior_a::Distribution = Normal(0., 1)
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2),
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	fit = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		model, 
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
	initial_params::Union{AbstractVector,Nothing}=nothing,
	prior_ρ::Union{Distribution, Missing},
	prior_a::Union{Distribution, Missing}
)
	model = single_p_QL(;
		N = nrow(data),
		n_blocks = maximum(data.block),
		block = data.block,
		valence = unique(data[!, [:block, :valence]]).valence,
		choice = data.choice,
		outcomes = hcat(
			data.feedback_suboptimal,
			data.feedback_optimal,
		),
		initV = fill(initV, 1, 2),
		prior_ρ = estimate == "MAP" ? prior_ρ : Normal(), ## For MLE, prior is meaningless
		prior_a = estimate == "MAP" ? prior_a : Normal()
	)

	if estimate == "MLE"
		fit = maximum_likelihood(model; initial_params = initial_params)
	elseif estimate == "MAP"
		fit = maximum_a_posteriori(model; initial_params = initial_params)
	end

	return fit
end

# Find MLE / MAP multiple times
function optimize_multiple_single_p_QL(
	data::DataFrame;
	initV::Float64,
	estimate::String = "MAP",
	include_true::Bool = false, # Whether to return true value if this is simulation
	prior_ρ::Union{Distribution, Missing},
	prior_a::Union{Distribution, Missing},
	initial_params::Union{AbstractVector,Nothing}=[ismissing(prior_ρ) ? 0. : mean(prior_ρ), ismissing(prior_a) ? 0. : mean(prior_a)],
)

	@assert (estimate == "MLE") || (!ismissing(prior_ρ) && !ismissing(prior_a)) "Must supply priors"

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
			initial_params = initial_params,
			prior_ρ = prior_ρ,
			prior_a = prior_a
		)

		# Return
		if include_true
			est = (
				PID = gdf.PID[1],
				true_a = α2a(gdf.α[1]),
				true_ρ = gdf.ρ[1],
				MLE_a = est.values[:a],
				MLE_ρ = est.values[:ρ]
				)
		else
			est = (
				PID = gdf.PID[1],
				a = est.values[:a],
				ρ = est.values[:ρ]
			)
		end

		lock(lk) do
			push!(ests, est)
		end
	end

	ests = DataFrame(ests)

	@assert sort(unique(ests.PID)) == sort(unique(data.PID))

	return sort(ests, :PID)
end

"""
bootstrap_optimize_single_p_QL(
    PLT_data::AbstractDataFrame;
    n_bootstraps::Int64 = 20,
    initV::Float64 = aao,
    prior_ρ::Distribution = truncated(Normal(0., 2.), lower = 0.),
    prior_a::Distribution = Normal()
) -> AbstractDataFrame

Bootstrap participant parameters from a Q-Learning model.

# Arguments
- `PLT_data::AbstractDataFrame`: The input data containing participant-level trial information. Must be a DataFrame or similar structure.
- `n_bootstraps::Int64`: The number of bootstrap samples to generate. Defaults to 20.
- `initV::Float64`: Initial value for the optimization procedure, with a default value `aao`.
- `prior_ρ::Distribution`: The prior distribution for the parameter `ρ`, defaulting to a truncated normal distribution `Normal(0., 2.)` with a lower bound of 0.
- `prior_a::Distribution`: The prior distribution for the parameter `a`, defaulting to a standard normal distribution `Normal()`.

# Returns
- `bootstraps::AbstractDataFrame`: A DataFrame containing the results of the optimization, with additional bootstrap indices for each sample.

# Description
The function first prepares the input `PLT_data` for fitting by transforming it as necessary. It then performs an optimization using the `optimize_multiple_single_p_QL` function, leveraging Maximum A Posteriori (MAP) estimation with the specified priors. After fitting, the results are joined with additional participant identifiers (`pids`) using an `innerjoin` on the `PID` column, adding extra information such as condition and prolific_pid.

The function then generates bootstrap samples by sampling the rows of the joined data with replacement, appending a `bootstrap_idx` column to indicate the bootstrap iteration. 

The returned DataFrame contains all bootstrap samples concatenated together, allowing for further analysis of the variability in the fit results.
"""
function bootstrap_optimize_single_p_QL(
	PLT_data::AbstractDataFrame;
	n_bootstraps::Int64 = 20,
	initV::Float64 = aao,
	prior_ρ::Distribution = truncated(Normal(0., 2.), lower = 0.),
	prior_a::Distribution = Normal(),
	estimate::String = "MAP",
	real_data::Bool = true
) 

	# Prepare data for fit
	if real_data
		forfit, pids = prepare_for_fit(PLT_data)
	else
		forfit = copy(PLT_data)
	end

	# Fit
	fit = optimize_multiple_single_p_QL(
		forfit;
		initV = initV,
		estimate = estimate,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	# Add condition and prolific_pid
	fit = innerjoin(fit, pids, on = :PID)

	# Sample participants and add bootstrap id
	bootstraps = vcat([insertcols(
		fit[sample(Xoshiro(i), 1:nrow(fit), nrow(fit), replace=true), :],
		:bootstrap_idx => i
	) for i in 1:n_bootstraps]...)

	return bootstraps

end

# Sample from posterior for multiple datasets drawn for prior and summarise for simulation-based calibration
function SBC_single_p_QL(
	data::DataFrame;
	initV::Float64,
	random_seed::Union{Int64, Nothing} = nothing,
	iter_sampling = 500,
	prior_ρ::Distribution,
	prior_a::Distribution
)

	sums = []
	for p in unique(data.PID)
		gdf = filter(x -> x.PID == p, data)
		
		draws = posterior_sample_single_p_QL(
			gdf;
			initV = initV,
			random_seed = random_seed,
			iter_sampling = iter_sampling,
			prior_ρ = prior_ρ,
			prior_a = prior_a
		)

		push!(
			sums,
			sum_SBC_draws(
				draws,
				params = [:a, :ρ],
				true_values = [α2a(gdf.α[1]), gdf.ρ[1]],
				prior_var = [1., var(truncated(Normal(0., 2.), lower = 0.))]
			)
		)
	end
	
	return sums
end

# Simulate choice from posterior 
function simulate_from_posterior_single_p_QL(
	bootstrap::DataFrameRow, # DataFrameRow containinng parameters and condition
	task, # Task structure for condition,
	random_seed::Int64
) 

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	block = task.block
	valence = task.valence
	outcomes = task.outcomes

	post_model = single_p_QL(
		N = length(block),
		n_blocks = maximum(block),
		block = block,
		valence = valence,
		choice = fill(missing, length(block)),
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		prior_ρ = Normal(-.99), # Prior doesn't matter in posterior simulation
		prior_a = Normal(-.99)
	)
	
	chn = Chains([bootstrap.a bootstrap.ρ], [:a, :ρ])

	choice = predict(Xoshiro(random_seed), post_model, chn)[1, :, 1] |> Array |> vec
	
	ppc = insertcols(task.task, 
		:isOptimal => choice,
		:bootstrap_idx => fill(bootstrap.bootstrap_idx, length(choice)),
		:prolific_pid => fill(bootstrap.prolific_pid, length(choice))
	)
	

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
