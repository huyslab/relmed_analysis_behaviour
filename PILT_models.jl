# PILT task Turing.jl models
# Each model should be folowed by a function mapping data DataFrame into arguments for the model

"""
    single_p_QL(; block::Vector{Int64}, valence::AbstractVector, 
                choice, outcomes::Matrix{Float64}, initV::Matrix{Float64}, 
                σ_ρ::Float64=1.0, σ_a::Float64=0.5)

Performs Q-learning for a single participant in a reinforcement learning task, with trial-wise updates of Q-values based on choices and outcomes.

# Arguments
- `block::Vector{Int64}`: A vector indicating the block number for each trial.
- `valence::AbstractVector`: A vector of valence values associated with each block, modulating the Q-value updates.
- `choice`: A binary vector representing the participant's choices (e.g., `true` for choosing stimulus A). Not typed to allow for both empirical and simulated data.
- `outcomes::Matrix{Float64}`: A matrix of outcomes for the options, where the first column corresponds to the suboptimal option and the second column to the optimal option.
- `initV::Matrix{Float64}`: Initial Q-values for the options, used as a starting point for learning.
- `σ_ρ::Float64=1.0`: Standard deviation for the prior distribution of the reward sensitivity parameter `ρ`.
- `σ_a::Float64=0.5`: Standard deviation for the prior distribution of the learning rate parameter `a`.

# Returns
- `Qs`: A matrix of updated Q-values for each trial, reflecting the participant's learning over time.

# Details
- The function models the Q-learning process where the Q-values (expected values) for each option are updated based on prediction errors.
- `ρ` is the reward sensitivity parameter, drawn from a truncated normal distribution, which scales the outcomes.
- `a` is the learning rate parameter, drawn from a normal distribution, transformed via a logistic function to compute the learning rate `α`.
- The Q-values are initialized using the provided `initV` matrix and are scaled by `ρ` and modulated by the `valence` of the block in which the trial occurs.
- On each trial, the participant's choice is modeled using a Bernoulli distribution, where the probability is determined by the difference in Q-values of the two options.
- The chosen option's Q-value is updated based on the prediction error (`PE`), which is the difference between the observed outcome and the expected value.
"""
@model function single_p_QL(;
	block::Vector{Int64}, # Block number
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values,
	prior_ρ::Distribution = truncated(Normal(0., 2.), lower = 0.),
	prior_a::Distribution = Normal(0., 1)
)

	# Priors on parameters
	ρ ~ prior_ρ
	a ~ prior_a

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
    map_data_to_single_p_QL(data::AbstractDataFrame) -> NamedTuple

Maps the given dataframe `data` into a named tuple containing the required arguments for the 
`single_p_QL` Turing model. The dataframe is first cleaned by dropping any missing values.

# Arguments
- `data::AbstractDataFrame`: A dataframe with columns `block`, `choice`, `feedback_suboptimal`, 
  and `feedback_optimal`.

# Returns
A named tuple with the following keys:
- `block`: A vector containing the `block` values.
- `choice`: A vector of `choice` values from the dataframe.
- `outcomes`: A matrix where each column contains the `feedback_suboptimal` and `feedback_optimal` 
   values from the dataframe.

This structure can be passed as arguments to the `single_p_QL` model.
"""
function map_data_to_single_p_QL(
	data::AbstractDataFrame
)

	tdata = dropmissing(data)

	return (;
		block = collect(tdata.block),
		choice = tdata.choice,
		outcomes = hcat(tdata.feedback_suboptimal, tdata.feedback_optimal)
	)

end

"""
    single_p_recip_QL(; block::Vector{Int64}, valence::AbstractVector, 
                      choice, outcomes::Matrix{Float64}, initV::Matrix{Float64}, 
                      σ_ρ::Float64=1.0, σ_a::Float64=0.5)

Performs a variant of Q-learning for a single participant in a reinforcement learning task, where Q-values are updated reciprocally for the chosen and unchosen options based on choices and outcomes.

# Arguments
- `block::Vector{Int64}`: A vector indicating the block number for each trial.
- `valence::AbstractVector`: A vector of valence values associated with each block, modulating the Q-value updates.
- `choice`: A binary vector representing the participant's choices (e.g., `true` for choosing stimulus A). Not typed to allow for both empirical and simulated data.
- `outcomes::Matrix{Float64}`: A matrix of outcomes for the options, where the first column corresponds to the suboptimal option and the second column to the optimal option.
- `initV::Matrix{Float64}`: Initial Q-values for the options, used as a starting point for learning.
- `σ_ρ::Float64=1.0`: Standard deviation for the prior distribution of the reward sensitivity parameter `ρ`.
- `σ_a::Float64=0.5`: Standard deviation for the prior distribution of the learning rate parameter `a`.

# Returns
- `Qs`: A matrix of updated Q-values for each trial, reflecting the participant's learning over time with reciprocal updates.

# Details
- The function models a modified Q-learning process where the Q-values (expected values) for both the chosen and unchosen options are updated based on prediction errors.
- `ρ` is the reward sensitivity parameter, drawn from a truncated normal distribution, which scales the outcomes.
- `a` is the learning rate parameter, drawn from a normal distribution, transformed via a logistic function to compute the learning rate `α`.
- The Q-values are initialized using the provided `initV` matrix and are scaled by `ρ` and modulated by the `valence` of the block in which the trial occurs.
- On each trial, the participant's choice is modeled using a Bernoulli distribution, where the probability is determined by the difference in Q-values of the two options.
- The chosen option's Q-value is updated based on the prediction error (`PE`), which is the difference between the observed outcome and the expected value.
- In this variant, the unchosen option's Q-value is reciprocally updated by subtracting the same `α * PE`, leading to a balance between the two options' values over time.
"""
@model function single_p_recip_QL(;
	block::Vector{Int64}, # Block number
	valence::AbstractVector, # Valence of each block
	choice, # Binary choice, coded true for stimulus A. Not typed so that it can be simulated
	outcomes::Matrix{Float64}, # Outcomes for options, second column optimal
	initV::Matrix{Float64}, # Initial Q values,
	σ_ρ::Float64 = 5.,
	σ_a::Float64 = 1.
)

	# Priors on parameters
	ρ ~ truncated(Normal(0., σ_ρ), lower = 0.)
	a ~ Normal(0., σ_a)

	# Compute learning rate
	α = a2α(a) # hBayesDM uses Phi_approx from Stan. Here, logistic with the variance of the logistic multiplying a to equate the scales to that of a probit function.

	# Initialize Q values
	Qs = repeat(initV .* ρ, length(block)) .* valence[block]

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

# Same mapping function as single_p_QL
map_data_to_single_p_recip_QL = map_data_to_single_p_QL