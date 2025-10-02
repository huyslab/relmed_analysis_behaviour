"""
Hierarchical Bayesian models for PILT project development.
"""

using Turing, Distributions, DynamicPPL, LinearAlgebra
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
    y::Union{Missing, Vector{Float64}},
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
    y::Union{Missing, Vector{Float64}},
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

    # Likelihood with hierarchical structure
    y ~ arraydist(BernoulliLogit.(μ .+ θ)[participant_id])
end

@model function hierarchical_logistic_regression_one_grouping(;
    y::Union{Missing, Vector{Float64}},
    X::Matrix{Float64}, # Model matrix, including intercept
    N_predictors::Int,
    participant_id::Vector{Int},
    N_participants::Int,
    priors::Dict = Dict(
            :β => filldist(Normal(0, 1), N_predictors),  # Coefficients for intercept and slope
            :τ => filldist(truncated(Normal(0, 1), 0, Inf), N_predictors),
            :η => 2.0  # LKJ correlation prior concentration
    )
)
    
    # Group-level hyperpriors
    β ~ priors[:β]  # Group mean
    τ ~ priors[:τ]  # Between-participant variability

    # Correlation structure for random effects
    Ω ~ LKJCholesky(N_predictors, priors[:η])  # Cholesky factor of correlation matrix

    # Participant-level random coefficients 
    z ~ filldist(MvNormal(zeros(N_predictors), I), N_participants)
    w = (Diagonal(τ) * Ω.L * z)'

    # Vectorized participant-specific predictions
    participant_effects = sum(X .* w[participant_id, :], dims=2)[:] # n_trials x n_predictors .* n_trials x n_predictors, summed over columns to give n_trials x 1
    m = X * β .+ participant_effects # n_trials x n_predictors * n_predictors x 1 + n_trials x 1

    # Likelihood with hierarchical structure
    y ~ arraydist(BernoulliLogit.(m))
end
