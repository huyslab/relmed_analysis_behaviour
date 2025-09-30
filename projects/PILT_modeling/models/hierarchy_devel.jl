using Turing, Distributions, DynamicPPL

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

    μ ~ priors[:μ]  # Hyperprior for the group mean
    σ_participant ~ priors[:σ_participant]  # Hyperprior for participant-level stddev
    σ ~ priors[:σ]  # Prior for observation noise

    θ ~ filldist(Normal(0, σ_participant), N_participants) # Varying intercepts for participants

    y ~ MvNormal(μ .+ θ[participant_id], I * σ)  # Likelihood
end
