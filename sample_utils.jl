# Functions for models and working with draws from models

# Simulate data from model prior
function simulate_single_p_PILT(
	n::Int64; # How many datasets to simulate
	model::Function,
	block::Vector{Int64}, # Block number
	outcomes::Matrix{Float64}, # Outcomes for options, first column optimal
	initV::Matrix{Float64}, # Initial Q values
	random_seed::Union{Int64, Nothing} = nothing,
	prior_ρ::Distribution,
	prior_a::Distribution
)

	# Total trial number
	N = length(block)

	# Trials per block
    n_trials = div(length(block), maximum(block))

	# Prepare model for simulation
	prior_model = model(
		N = N,
		n_blocks = maximum(block),
		block = block,
		choice = fill(missing, length(block)),
		outcomes = outcomes,
		initV = initV,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	# Draw parameters and simulate choice
	prior_sample = sample(
		isnothing(random_seed) ? Random.default_rng() : Xoshiro(random_seed),
		prior_model,
		Prior(),
		n
	)

	# Arrange choice for return
	sim_data = DataFrame(
		PID = repeat(1:n, inner = N),
		ρ = repeat(prior_sample[:, :ρ, 1], inner = N),
		a = repeat(prior_sample[:, :a, 1], inner = N),
		α = repeat(prior_sample[:, :a, 1], inner = N) .|> a2α,
		block = repeat(block, n),
		valence = repeat(sign.(outcomes[:,1]), outer = n),
		trial = repeat(1:n_trials, n * maximum(block)),
		choice = prior_sample[:, [Symbol("choice[$i]") for i in 1:N], 1] |>
			Array |> transpose |> vec
	)

	# Compute Q values
	Qs = generated_quantities(prior_model, prior_sample) |> vec

	sim_data.Q_optimal = vcat([qs[:, 2] for qs in Qs]...) 
	sim_data.Q_suboptimal = vcat([qs[:, 1] for qs in Qs]...) 

	return sim_data
			
end


# Transform unconstrainted a to learning rate α
a2α(a) = logistic(π/sqrt(3) * a)

α2a(α) = logit(α) / (π/sqrt(3))

@assert α2a(a2α(0.5)) ≈ 0.5

# Compute posterior quantiles
begin
	lb(x) = quantile(x, 0.25)
	ub(x) = quantile(x, 0.75)
	llb(x) = quantile(x, 0.025)
	uub(x) = quantile(x, 0.975)
end

# Exctract vector from chains
function single_layer_chain_to_vector(
    chain,
    param::String
)   
    # Find columns
    matching_columns = filter(x -> occursin(Regex("$(param)\\[\\d+\\]"), x), string.(names(chain, :parameters)))

    # Extract values
    outcome =  [vec(Array(chain[i, matching_columns, :])) for i in 1:size(chain, 1)]

    # Check lengths match
    @assert allequal(length.(outcome))

    # Reutrn flat if 1 iteration only
    if length(outcome) == 1
        return outcome[1]
    else
        return outcome
    end
end


# Summarize prior predictive draws relative to true value
function sum_SBC_draws(
	draws;
	params::Vector{Symbol}, # Parameters to sum
	true_values::Vector{Float64},
	prior_var::Vector{Float64} = repeat([1.], length(params)) # Variance of the prior for computing posterior contraction
	)

	if length(size(draws)) > 2 # For MCMC Chains object
		tdraws = DataFrame(Array(draws), names(draws, :parameters))
	else
		tdraws = draws
	end

	sums = []
	for (p, t, pv) in zip(params, true_values, prior_var)

		v = tdraws[!, p]
		v_s = v .- t

		push!(sums,
			(;
				Symbol("$(p)_m") => median(v), # Median posterior
				Symbol("$(p)_lb") => lb(v), # Posterior 25th percentile
				Symbol("$(p)_ub") => ub(v), # Posterior 75th percentile
				Symbol("$(p)_zs") => mean(v_s) / std(v), # Posterior zscore
				Symbol("$(p)_sm") => median(v_s), # Median error
				Symbol("$(p)_slb") => lb(v_s), # Error 25th percentile
				Symbol("$(p)_sub") => ub(v_s), # Error 75th percentile
				Symbol("$(p)_cntrct") => 1 - var(v) / pv, # Posterior contraction, after Schad et al. 2021
				Symbol("true_$p") => t
			)		
		)

	end

	# Return as one NamedTuple
	return reduce((x, y) -> merge(x, y), sums)

end

# Join two fit objects that used potentially different PID indices
function join_split_fits(
	fit1::DataFrame,
	fit2::DataFrame,
	pids1::DataFrame,
	pids2::DataFrame
)

	# Join
	fit1 = innerjoin(fit1, pids1, on = :PID)
	
	fit2 = innerjoin(fit2, pids2, on = :PID)
	
	fits = innerjoin(
		fit1[!, Not(:PID)], 
		fit2[!, Not(:PID)], 
		on = [:prolific_pid],
		renamecols = "_1" => "_2"
	)

	return fits
end