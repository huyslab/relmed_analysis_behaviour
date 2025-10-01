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
- `priors::Dict`: Prior distributions (μ, τ, σ)

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
            :τ => truncated(Normal(0, 1), 0, Inf),
            :σ => truncated(Normal(0, 1), 0, Inf)
        )
)
    # Group-level hyperpriors
    μ ~ priors[:μ]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability
    σ ~ priors[:σ]  # Within-participant (observation) noise

    # Participant-level random intercepts
    θ ~ filldist(Normal(0, τ), N_participants)

    # Likelihood with hierarchical structure
    y ~ MvNormal(μ .+ θ[participant_id], I * σ)
end

@model function simple_hierarchical_bernoulli_logit(;
    y,
    participant_id::Vector{Int},
    N_participants::Int,
    priors::Dict = Dict(
            :μ => Normal(0, 1),
            :τ => truncated(Normal(0, 1), 0, Inf)
        )
)
    # Group-level hyperpriors
    μ ~ priors[:μ]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability

    # Participant-level random intercepts
    θ ~ filldist(Normal(0, τ), N_participants)

    ϕ = logistic.(μ .+ θ)

    # Likelihood with hierarchical structure
    y ~ arraydist(Bernoulli.(ϕ)[participant_id])
end

@model function one_way_hierarchical_logistic_regression(;
    y,
    X::Matrix{Float64}, # Model matrix, including intercept
    participant_id::Vector{Int},
    N_participants::Int,
    priors::Dict = Dict(
            :β => filldist(Normal(0, 1), size(X, 2)),  # Coefficients for intercept and slope
            :τ => filldist(truncated(Normal(0, 1), 0, Inf), size(X, 2))
    )
)
    
    # Group-level hyperpriors
    β ~ priors[:β]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability

    # Participant-level random coefficients (participants in rows, coefficients in columns)
    b ~ filldist(Normal(0, τ), N_participants, size(X, 2))

    # Vectorized participant-specific predictions
    participant_effects = sum(X .* b[participant_id, :], dims=2)[:] # n_trials x n_predictors .* n_trials x n_predictors, summed over columns to give n_trials x 1
    ϕ = logistic.(X * β .+ participant_effects) # n_trials x n_predictors * n_predictors x 1 + n_trials x 1

    # Likelihood with hierarchical structure
    y ~ arraydist(Bernoulli.(ϕ))
end
