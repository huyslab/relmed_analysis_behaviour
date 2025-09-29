# Simple hierarchical regression models
begin
    using Turing, DynamicPPL, Distributions, Random
    include("$(pwd())/core/model_utils.jl")

    script_dir = dirname(@__FILE__)
    include(joinpath(script_dir, "..", "utils", "plotting.jl"))
end


@model function normal_intercept(;
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

chain, ground_truth = let N_participants = 50, N_obs = 200, ground_truth = Dict(
            :μ => Dirac(0.3),
            :σ_participant => Dirac(0.1),
            :σ => Dirac(0.05)
        )
    participant_id = repeat(1:N_participants, inner=N_obs)

    y = prior_sample(
        (;
            y = missing,
            participant_id = participant_id,
            N_participants = N_participants
        );
        model = normal_intercept,
        priors = ground_truth,
        outcome_name = :y,
        rng = Xoshiro(1)
    )

    data_model = normal_intercept(
        ;
        y = y,
        participant_id = participant_id,
        N_participants = N_participants
    )

    chain = sample(Xoshiro(0), data_model, NUTS(), MCMCThreads(), 1000, 4)

    chain, ground_truth

end

let
    f = Figure()

    plot_hyper_parameters_vs_fitted!(
        f,
        chain,
        ground_truth
    )

    f

end

