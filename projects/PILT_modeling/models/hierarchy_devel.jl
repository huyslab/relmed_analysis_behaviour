"""
Hierarchical Bayesian models for PILT project development.
"""

using Turing, Distributions, DynamicPPL
using StatsFuns: logistic

"""
    simple_hierarchical_normal(; y, participant_id, N_participants, priors)

Simple hierarchical normal model with participant-level random intercepts.

# Arguments
- `y`: Observed data vector
- `participant_id::Vector{Int}`: Participant indices for each observation
- `N_participants::Int`: Total number of participants
- `priors::Dict`: Prior distributions (μ, σ_participant, σ)

# Model structure
- Group-level mean μ with participant-specific deviations θ
- Hierarchical variance components for participants and observations
"""
@model function simple_hierarchical_normal(;
    y,
    participant_id::Vector{Int},
    N_participants::Int,
    priors::Dict = Dict(
            :μ => Normal(0, 1),
            :σ_participant => truncated(Normal(0, 1), 0, Inf),
            :σ => truncated(Normal(0, 1), 0, Inf)
        )
)
    # Group-level hyperpriors
    μ ~ priors[:μ]  # Group mean
    σ_participant ~ priors[:σ_participant]  # Between-participant variability
    σ ~ priors[:σ]  # Within-participant (observation) noise

    # Participant-level random intercepts
    θ ~ filldist(Normal(0, σ_participant), N_participants)

    # Likelihood with hierarchical structure
    y ~ MvNormal(μ .+ θ[participant_id], I * σ)
end

@model function simple_hierarchical_bernoulli_logit(;
    y,
    participant_id::Vector{Int},
    N_participants::Int,
    priors::Dict = Dict(
            :μ => Normal(0, 1),
            :σ_participant => truncated(Normal(0, 1), 0, Inf)
        )
)
    # Group-level hyperpriors
    μ ~ priors[:μ]  # Group mean
    σ_participant ~ priors[:σ_participant]  # Between-participant variability

    # Participant-level random intercepts
    θ ~ filldist(Normal(0, σ_participant), N_participants)

    ϕ = logistic.(μ .+ θ)

    # Likelihood with hierarchical structure
    y ~ arraydist(Bernoulli.(ϕ)[participant_id])
end
