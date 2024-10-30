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
- `choice`: Binary vector of choices for each trial (`true` for choosing option A). Not typed to allow empirical or simulated values.
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
	initial_Q::Union{Nothing, Float64} = nothing, # Initial Q values,
	priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    )
)

    # initial values
    initial_Q = isnothing(initial_Q) ? mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) : initial_Q
    initV::AbstractArray{Float64} = fill(initial_Q, 1, 2)

	# Priors on parameters
	ρ ~ priors[:ρ]
	a ~ priors[:a]

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values, with sign depending on block valence
	Qs = repeat(initV .* ρ, length(block)) .* sign.(outcomes[:, 1])

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

	# Drop missing data
	tdata = dropmissing(data[!, collect(values(columns))])

	# Sort
	sort!(tdata, [columns["block"], columns["trial"]])

	return (;
		block = collect(tdata[!, columns["block"]]),
		choice = tdata[!, columns["choice"]],
		outcomes = hcat(tdata[!, columns["feedback_suboptimal"]], tdata[!, columns["feedback_optimal"]])
	)
end

