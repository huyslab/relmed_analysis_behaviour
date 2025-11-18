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

function running_average_update(
    trial::Int,
    outcome::Float64,
    ρ::Real,
    Q::Real
)
    α = 1 / trial
    PE = outcome * ρ - Q
    return α * PE
end


function running_average_block_ll(;
    ρ::Real,
    N_trials::Int,
    choice::Vector{Int},
    trial::Vector{Int64}, # Trial number in block
    outcomes::Matrix{Float64},
    Q::Vector{<:Real}
)

    ll = 0.0
    for i in 1:N_trials
        # Likelihood at current Q
        ll += logpdf(Categorical(softmax(Q)), choice[i])

        # Update Q values
        Q[choice[i]] += running_average_update(
            trial[i],
            outcomes[i, choice[i]],
            ρ,
            Q[choice[i]]
        )
    end

    return ll
end

@model function hierarchical_running_average_blockloop(;
    block_starts::Vector{Int},
    block_ends::Vector{Int},
    trial::Vector{Int64}, # Trial number in block
    outcomes::Matrix{Float64},            # Outcomes for options (rows align with rows of block/trial/participant)
    N_actions::Int = 2,                   # Number of actions
    choice::Union{Vector{Missing}, Vector{Int}},
    participant_per_block::Vector{Int},
    N_participants::Int,
    initial_value::Float64 = 0.0,         # Initial Q values
    priors::Dict = Dict(
        :logρ => Normal(0, 1.5),
        :τ    => truncated(Normal(0, 0.5), 0, 100),
    ),
)
    # Group-level hyperpriors
    logρ ~ priors[:logρ]
    τ    ~ priors[:τ]

    # Participant-level random coefficients
    θ  ~ filldist(Normal(0, 1), N_participants)
    ρs = exp.(clamp.(logρ .+ τ .* θ, -10, 10))  # Individual reward sensitivities

    # Initialize Q outside the loop
    Q = fill(initial_value * ρs[1], N_actions)

    # Loop by block; sequential updates within a block
    if !any(ismissing.(choice))
        for bi in eachindex(block_starts)

            block_idx = block_starts[bi]:block_ends[bi]

            ll = running_average_block_ll(
                ρ = ρs[participant_per_block[bi]],
                N_trials = length(block_idx),
                choice = choice[block_idx],
                trial = trial[block_idx],
                outcomes = outcomes[block_idx, :],
                Q = copy(Q)  # Pass a copy to avoid mutating across blocks
            )

            @addlogprob! (; loglikelihood = ll)
            
        end
    else
        for bi in eachindex(block_starts)

            let Q = copy(Q)  # Local Q for this block
                block_idx = block_starts[bi]:block_ends[bi]
                ρ = ρs[participant_per_block[bi]]

                for idx in block_idx
                    # Draw/condition choice at current Q
                    choice[idx] ~ Categorical(softmax(Q))

                    # Update Q values
                    a = choice[idx]
                    r = outcomes[idx, a]
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

