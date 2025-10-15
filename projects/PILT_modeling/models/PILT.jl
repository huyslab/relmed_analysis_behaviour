using LogExpFunctions: softmax
using Distributions, Turing, DynamicPPL, LinearAlgebra
using Distributions: Categorical

@model function hierarchical_running_average(;
    block::Vector{Int64}, # Block number
	trial::Vector{Int64}, # Trial number in block
	outcomes::Matrix{Float64}, # Outcomes for options
    N_actions::Int = 2, # Number of actions
    choice::Union{Vector{Missing}, Vector{Int}},
    participant::Vector{Int},
    N_participants::Int,
    initial_value::Float64 = 0.0, # Initial Q values,
    priors::Dict = Dict(
            :logρ => Normal(0, 1.5),  
            :τ => truncated(Normal(0, 0.5), 0, 100),
    )
)
    
    # Group-level hyperpriors
    logρ ~ priors[:logρ]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability

    # Participant-level random coefficients 
    θ ~ filldist(Normal(0, 1), N_participants)

    ρs = exp.(clamp.(logρ .+ τ .* θ, -10, 10))  # Individual reward sensitivites

    # Initialize Q values
    Q = fill(initial_value * ρs[1], N_actions)  # Initialize, and set type to the same as ρs

    for i in eachindex(trial)

        if trial[i] == 1
            Q .= initial_value * ρs[participant[i]]  # Initialize Q values
        end

        α = 1 / trial[i]
        
        # Likelihood
        choice[i] ~ Categorical(softmax(Q))


        # Update Q values
        if (i < length(block) && (block[i] == block[i + 1]) && (participant[i] == participant[i + 1]))
           # Prediction error
		    PE = outcomes[i, choice[i]] * ρs[participant[i]] - Q[choice[i]]
           
            Q[choice[i]] += α * PE
        end

    end

end

@model function hierarchical_running_average2(;
    N_actions::Int = 2, # Number of actions
	outcomes::Vector{Matrix{Float64}}, # Outcomes for options
    choice::Union{Vector{Vector{Missing}}, Vector{Vector{Int}}},
    N_participants::Int,
    participant_per_block::Vector{Int},
    initial_value::Float64 = 0.0, # Initial Q values,
    priors::Dict = Dict(
            :logρ => Normal(0, 1.5),  
            :τ => truncated(Normal(0, 0.25), 0, 100),
    )
)
    
    # Group-level hyperpriors
    logρ ~ priors[:logρ]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability

    # Participant-level random coefficients 
    θ ~ filldist(Normal(0, 1), N_participants)

    ρs = exp.(clamp.(logρ .+ τ .* θ, -5, 5))  # Individual reward sensitivites

    # Run over blocks
    for b in eachindex(outcomes)
        Q = fill(initial_value * ρs[participant_per_block[b]], N_actions)  # Initialize

        # Run over trials in block
        for t in eachindex(choice[b])

            # Learning rate
            α = 1 / t

            # Likelihood
            choice[b][t] ~ Categorical(softmax(Q))

            # Prediction error
            PE = outcomes[b][t, choice[b][t]] * ρs[participant_per_block[b]] - Q[choice[b][t]]

            # println("Block: $b, Trial: $t, Choice: $(choice[b][t]), outcome: $(outcomes[b][t, choice[b][t]]), reward: $(outcomes[b][t, choice[b][t]] * ρs[participant_per_block[b]]), PE: $PE Q: $Q, ρ: $(ρs[participant_per_block[b]]), p: $(softmax(Q))")


            # Update Q values
            Q[choice[b][t]] += α * PE

            Q .= clamp.(Q, -50, 50)
        end

    end
end

