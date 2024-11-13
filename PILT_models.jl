# PILT task Turing.jl models
# Each model should be folowed by a function mapping data DataFrame into arguments for the model

"""
    single_p_QL(;
        block::Vector{Int64}, 
        outcomes::Matrix{Float64}, 
        choice, 
        initV::Union{Nothing, Float64} = nothing, 
        priors::Dict = Dict(
            :ρ => truncated(Normal(0., 1.), lower = 0.),
            :a => Normal(0., 0.5)
        )
    )

Defines a Turing model for single-participant Q-learning in a reinforcement learning task, updating Q-values based on choice outcomes across trials and blocks.

# Arguments
- `block::Vector{Int64}`: Indicates the block number of each trial.
- `outcomes::Matrix{Float64}`: Matrix with two columns, where the first represents suboptimal outcomes and the second represents optimal outcomes.
- `choice`: Binary vector of choices for each trial (`true` for choosing optimal stimulus). Not typed to allow empirical or simulated values.
- `initV::Union{Nothing, Float64} = nothing`: Initial Q-value or `nothing`, which will be set to a mean value if unspecified.
- `priors::Dict`: Dictionary of prior distributions for parameters:
  - `ρ`: Reward sensitivity parameter, scaling outcomes.
  - `a`: Learning rate parameter, transformed to compute the learning rate `α`.

# Returns
- `Qs`: Matrix of updated Q-values for each trial, showing the participant's learning across time.

# Details
- Initializes Q-values based on `initV` and scales them by `ρ`.
- Computes learning rate `α` via a logistic transformation of `a` for stability.
- For each trial, updates the Q-values using a prediction error (`PE`), calculated as the difference between the observed outcome and the current Q-value.
- Updates are applied sequentially for each trial within a block, with choices modeled as a Bernoulli distribution based on the difference in Q-values between options.
- Designed for flexible specification of priors and initial Q-value setup.
"""
@model function single_p_QL(;
	block::Vector{Int64}, # Block number
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	initV::Union{Nothing, Float64} = nothing, # Initial Q values,
	priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    )
)

    # initial values
    initV = isnothing(initV) ? mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) : initV
    initial_Q::AbstractArray{Float64} = fill(initV, 1, 2)

	# Priors on parameters
	ρ ~ priors[:ρ]
	a ~ priors[:a]

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values, with sign depending on block valence
	Qs = repeat(initial_Q .* ρ, length(block)) .* sign.(outcomes[:, 1])

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:length(block)
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 2] - Qs[i, 1])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != length(block)) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx]
		end
	end

	return Qs

end

function check_PILT_data(
	data::DataFrame;
	columns::Dict
)

	@assert all(.![eltype(col) isa Union && Missing in Base.uniontypes(eltype(col)) for col in values(columns)]) "Missing values allowed in DataFrame columns"

	@assert issorted(data[!, columns["block"]]) "Data not sorted by block number"

	@assert all(combine(groupby(data, columns["block"]), columns["trial"] => issorted => :ok).ok) "Data not sorted by trial number"

end

"""
    unpack_single_p_QL(data::AbstractDataFrame; columns::Dict{String, Symbol} = default_columns) -> NamedTuple

Transforms the given dataframe `data` into a named tuple with the required structure for the 
`single_p_QL` Turing model, using the specified column mappings. Missing values are dropped, and the 
data is sorted by `block` and `trial`.

# Arguments
- `data::AbstractDataFrame`: A dataframe containing columns for `block`, `trial`, `feedback_suboptimal`, 
  `feedback_optimal`, and `choice` or the chosen mapping for each column.
- `columns::Dict{String, Symbol}`: A dictionary that maps expected data keys (e.g., `"block"`, `"trial"`, 
  `"feedback_optimal"`, `"feedback_suboptimal"`, `"choice"`) to column names in `data`. Defaults to a set 
  of common column names.

# Returns
A named tuple with the following keys:
- `block`: A vector of `block` values, representing the block structure of trials.
- `choice`: A vector containing values indicating whether each choice was optimal.
- `outcomes`: A matrix where each column contains the `feedback_suboptimal` and `feedback_optimal` values 
  for each trial.

This structure is formatted for compatibility with the `single_p_QL` model.
"""
function unpack_single_p_QL(
	data::AbstractDataFrame;
	columns::Dict{String, Symbol} = Dict(
		"block" => :block,
		"trial" => :trial,
		"feedback_optimal" => :feedback_optimal,
		"feedback_suboptimal" => :feedback_suboptimal,
		"choice" => :isOptimal
	)
)

	check_PILT_data(data; columns)

	return (;
		block = data[!, columns["block"]],
		choice = data[!, columns["choice"]],
		outcomes = hcat(data[!, columns["feedback_suboptimal"]], data[!, columns["feedback_optimal"]])
	)
end


"""
    single_p_QL_recip(; 
        block::Vector{Int64}, 
        outcomes::Matrix{Float64}, 
        choice, 
        initial_Q::Union{Nothing, Float64} = nothing, 
        priors::Dict = Dict(
            :ρ => truncated(Normal(0., 1.), lower = 0.),
            :a => Normal(0., 0.5)
        )
    )

A Q-learning model with reciprocal updating, built in Turing, for a two-choice task. This model includes parameters for a learning rate and reward sensitivity. Reciprocal updates are applied such that an increase in the chosen option's Q-value results in a commensurate decrease for the unchosen one.

# Arguments
- `block::Vector{Int64}`: Vector of block numbers.
- `outcomes::Matrix{Float64}`: A matrix of feedback, where each row corresponds to a trial. The first column represents outcomes for suboptimal option, and the second for the optimal option.
- `choice`: Binary choices per trial. True indicates selection of optimal stimulus.
- `initial_Q::Union{Nothing, Float64} = nothing`: Initial Q-value for both options. Defaults to a mean-centered value if `nothing` is provided.
- `priors::Dict`: Dictionary of prior distributions for parameters. Default priors are:
    - `:ρ`: A truncated normal prior (mean = 0, sd = 1, lower bound = 0).
    - `:a`: A normal prior (mean = 0, sd = 0.5).

# Model Description
1. **Initialize Q-values**: Sets initial Q-values to `initial_Q` or a centered default if not specified.
2. **Sample Parameters**:
   - `ρ`: Reward sensitivity.
   - `a`: Parameter determining the learning rate, sampled from its prior.
   - `α`: Learning rate, calculated using a logistic transformation of `a`.
3. **Q-value Updates**:
   - Q-values are initialized and scaled by `ρ`, and are signed based on the block valence, assumed to be constant.
   - On each trial, the choice distribution is specified as a Bernoulli distribution, driven by the difference in Q-values.
   - For the chosen option, the prediction error (`PE`) is computed and used to update Q-values.
   - Reciprocal updating is applied such that the unchosen option's Q-value is inversely adjusted by `α * PE` for future trials within the same block.

# Returns
- `Qs`: Matrix of Q-values after updating across trials.
"""
@model function single_p_QL_recip(;
	block::Vector{Int64}, # Block number
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	initV::Union{Nothing, Float64} = nothing, # Initial Q values,
	priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    )
)

    # initial values
    initV = isnothing(initV) ? mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) : initV
    initial_Q::AbstractArray{Float64} = fill(initV, 1, 2)

	# Priors on parameters
	ρ ~ priors[:ρ]
	a ~ priors[:a]

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values, with sign depending on block valence
	Qs = repeat(initial_Q .* ρ, length(block)) .* sign.(outcomes[:, 1])

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:length(block)
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 2] - Qs[i, 1])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != length(block)) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx] - α * PE
		end
	end

	return Qs

end

@model function running_average(;
	block::Vector{Int64}, # Block number
	trial::Vector{Int64}, # Trial number in block
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	initV::Union{Nothing, Float64} = nothing, # Initial Q values,
	priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.)
    )
)

    # initial values
    initV = isnothing(initV) ? mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) : initV
    initial_Q::AbstractArray{Float64} = fill(initV, 1, 2)

	# Priors on parameters
	ρ ~ priors[:ρ]

	# Initialize Q values, with sign depending on block valence
	Qs = repeat(initial_Q .* ρ, length(block)) .* sign.(outcomes[:, 1])

	# Loop over trials, updating Q values and incrementing log-density
	for i in 1:length(block)

		# Learning rate is 1/N
		α = 1 / trial[i]
		
		# Define choice distribution
		choice[i] ~ BernoulliLogit(Qs[i, 2] - Qs[i, 1])

		choice_idx::Int64 = choice[i] + 1

		# Prediction error
		PE = outcomes[i, choice_idx] * ρ - Qs[i, choice_idx]

		# Update Q value
		if (i != length(block)) && (block[i] == block[i+1])
			Qs[i + 1, choice_idx] = Qs[i, choice_idx] + α * PE
			Qs[i + 1, 3 - choice_idx] = Qs[i, 3 - choice_idx] - α * PE
		end
	end

	return Qs

end

function unpack_running_average(
	data::AbstractDataFrame;
	columns::Dict{String, Symbol} = Dict(
		"block" => :block,
		"trial" => :trial,
		"feedback_optimal" => :feedback_optimal,
		"feedback_suboptimal" => :feedback_suboptimal,
		"choice" => :isOptimal
	)
)

	check_PILT_data(data; columns)

	return (;
		block = data[!, columns["block"]],
		trial = data[!, columns["trial"]],
		choice = data[!, columns["choice"]],
		outcomes = hcat(data[!, columns["feedback_suboptimal"]], data[!, columns["feedback_optimal"]])
	)
end
