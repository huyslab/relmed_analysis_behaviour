using LogExpFunctions: softmax
using Distributions, Turing, DynamicPPL, LinearAlgebra
using Distributions: Categorical

"""
Hierarchical running average model for PILT task.
Uses a trial-by-trial loop with individual Q-value updates.

# Arguments
- `block`: Block number for each trial
- `trial`: Trial number within each block (resets to 1 at start of each block)
- `outcomes`: Matrix of outcomes (trials × actions)
- `N_actions`: Number of available actions/options (default: 2)
- `choice`: Vector of choices (can contain Missing for prediction)
- `participant`: Participant ID for each trial
- `N_participants`: Total number of participants
- `initial_value`: Initial Q-value before scaling by ρ (default: 0.0)
- `priors`: Dictionary of prior distributions for group-level parameters
"""
@model function hierarchical_running_average(;
    block::Vector{Int64},
	trial::Vector{Int64},
	outcomes::Matrix{Float64},
    N_actions::Int = 2,
    choice::Union{Vector{Missing}, Vector{Int}},
    participant::Vector{Int},
    N_participants::Int,
    initial_value::Float64 = 0.0,
    priors::Dict = Dict(
            :logρ => Normal(0, 1.5),  
            :τ => truncated(Normal(0, 0.5), 0, 100),
    )
)
    
    # Group-level hyperpriors for hierarchical model
    logρ ~ priors[:logρ]  # Group mean of log reward sensitivity
    τ ~ priors[:τ]  # Between-participant standard deviation (variability)

    # Participant-level random effects (centered parameterization)
    θ ~ filldist(Normal(0, 1), N_participants)  # Standard normal deviates

    # Transform to individual reward sensitivities via non-centered parameterization
    # Clamping prevents extreme values that could cause numerical issues
    ρs = exp.(clamp.(logρ .+ τ .* θ, -10, 10))

    # Initialize Q values for all actions
    # Type is set to match ρs for type stability
    Q = fill(initial_value * ρs[1], N_actions)

    # Loop through all trials sequentially
    for i in eachindex(trial)

        # Reset Q values at the start of each block
        if trial[i] == 1
            Q .= initial_value * ρs[participant[i]]
        end

        # Running average: learning rate decreases as 1/trial_number
        α = 1 / trial[i]
        
        # Likelihood: choice is drawn from softmax over Q values
        choice[i] ~ Categorical(softmax(Q))

        # Update Q values based on observed outcome
        # Only update if we're not at the end of a block/participant sequence
        if (i < length(block) && (block[i] == block[i + 1]) && (participant[i] == participant[i + 1]))
            # Compute prediction error (reward scaled by sensitivity minus expected value)
		    PE = outcomes[i, choice[i]] * ρs[participant[i]] - Q[choice[i]]
           
            # Update Q value for chosen action using running average rule
            Q[choice[i]] += α * PE
        end

    end

end

"""
Compute the Q-value update for a single trial using running average rule.

# Arguments
- `trial`: Trial number within block (determines learning rate α = 1/trial)
- `outcome`: Observed outcome/reward
- `ρ`: Individual reward sensitivity parameter
- `Q`: Current Q-value for the chosen action

# Returns
- Q-value update (Δ Q)
"""
function running_average_update(
    trial::Int,
    outcome::Float64,
    ρ::Real,
    Q::Real
)
    # Learning rate decreases over trials (running average)
    α = 1 / trial
    # Prediction error: scaled reward minus current expectation
    PE = outcome * ρ - Q
    # Return the update increment
    return α * PE
end


"""
Compute log-likelihood for a complete block using running average updates.
This function is used for efficient likelihood computation when choices are observed.

# Arguments
- `ρ`: Reward sensitivity parameter for this participant
- `N_trials`: Number of trials in the block
- `choice`: Vector of choices made in this block
- `trial`: Trial numbers within block
- `outcomes`: Matrix of outcomes for this block
- `Q`: Initial Q-values (will be modified in place)

# Returns
- Total log-likelihood for the block
"""
function running_average_block_ll(;
    ρ::Real,
    N_trials::Int,
    choice::Vector{Int},
    trial::Vector{Int64},
    outcomes::Matrix{Float64},
    Q::Vector{<:Real}
)

    ll = 0.0
    for i in 1:N_trials
        # Add log probability of observed choice given current Q values
        ll += logpdf(Categorical(softmax(Q)), choice[i])

        # Update Q values for next trial
        Q[choice[i]] += running_average_update(
            trial[i],
            outcomes[i, choice[i]],
            ρ,
            Q[choice[i]]
        )
    end

    return ll
end

"""
Hierarchical running average model using block-wise loops for efficiency.
Processes data block-by-block rather than trial-by-trial.

# Arguments
- `block_starts`: Indices where each block starts
- `block_ends`: Indices where each block ends
- `trial`: Trial number within each block
- `outcomes`: Matrix of outcomes (trials × actions)
- `N_actions`: Number of available actions (default: 2)
- `choice`: Vector of choices (can contain Missing for prediction)
- `participant_per_block`: Participant ID for each block
- `N_participants`: Total number of participants
- `initial_value`: Initial Q-value before scaling (default: 0.0)
- `priors`: Dictionary of prior distributions
"""
@model function hierarchical_running_average_blockloop(;
    block_starts::Vector{Int},
    block_ends::Vector{Int},
    trial::Vector{Int64},
    outcomes::Matrix{Float64},
    N_actions::Int = 2,
    choice::Union{Vector{Missing}, Vector{Int}},
    participant_per_block::Vector{Int},
    N_participants::Int,
    initial_value::Float64 = 0.0,
    priors::Dict = Dict(
        :logρ => Normal(0, 1.5),
        :τ    => truncated(Normal(0, 0.5), 0, 100),
    ),
)
    # Group-level hyperpriors
    logρ ~ priors[:logρ]  # Group mean of log reward sensitivity
    τ    ~ priors[:τ]     # Between-participant variability

    # Participant-level random effects
    θ  ~ filldist(Normal(0, 1), N_participants)
    # Individual reward sensitivities (non-centered parameterization with clamping)
    ρs = exp.(clamp.(logρ .+ τ .* θ, -10, 10))

    # Initialize Q values (type matches ρs for stability)
    Q = fill(initial_value * ρs[1], N_actions)

    # Branch based on whether we're fitting (no missing) or predicting (has missing)
    if !any(ismissing.(choice))
        # Fitting mode: all choices observed, use efficient block likelihood
        for bi in eachindex(block_starts)

            # Get indices for current block
            block_idx = block_starts[bi]:block_ends[bi]

            # Compute log-likelihood for entire block
            ll = running_average_block_ll(
                ρ = ρs[participant_per_block[bi]],
                N_trials = length(block_idx),
                choice = choice[block_idx],
                trial = trial[block_idx],
                outcomes = outcomes[block_idx, :],
                Q = copy(Q)  # Use fresh Q values for each block
            )

            # Add block log-likelihood to model total
            @addlogprob! (; loglikelihood = ll)
            
        end
    else
        # Prediction mode: some choices missing, need trial-by-trial sampling
        for bi in eachindex(block_starts)

            # Use local Q values to avoid cross-block contamination
            let Q = copy(Q)
                block_idx = block_starts[bi]:block_ends[bi]
                ρ = ρs[participant_per_block[bi]]

                # Process each trial in block sequentially
                for idx in block_idx
                    # Sample or condition on choice given current Q values
                    choice[idx] ~ Categorical(softmax(Q))

                    # Update Q values based on chosen action and outcome
                    a = choice[idx]  # Action taken
                    r = outcomes[idx, a]  # Reward received
                    Q[a] += running_average_update(
                        trial[idx],
                        r,
                        ρ,
                        Q[a]
                    )
                end
            end
        end

    end
end

