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


