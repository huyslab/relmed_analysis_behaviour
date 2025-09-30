# Simple hierarchical regression models
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()


    using Turing, DynamicPPL, Distributions, Random, DataFrames
    include("$(pwd())/core/model_utils.jl")

    script_dir = dirname(@__FILE__)
    include(joinpath(script_dir, "..", "utils", "plotting.jl"))
    include(joinpath(script_dir, "..", "config.jl"))
end


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

function extract_vector_parameter(
    chain::Chains,
    parameter::String
)   
    regex = Regex("^$(parameter)\\[\\d+\\]\$")
    param_names = filter(name -> occursin(regex, name), string.(names(chain)))
    df = DataFrame(chain[:, Symbol.(param_names), :])
    return stack(df, Not([:iteration, :chain]); variable_name = :parameter)
end

chain, ground_truth, theta = 
    let N_participants = 50, N_obs = 200, ground_truth = Dict(
            :μ => Dirac(0.3),
            :σ_participant => Dirac(0.1),
            :σ => Dirac(0.05)
        )
    participant_id = repeat(1:N_participants, inner=N_obs)

    missing_data_model = simple_hierarchical_normal(
        ;
        y = missing,
        participant_id = participant_id,
        N_participants = N_participants,
        priors = ground_truth
    )

    prior_draw = sample(
        Xoshiro(1),
        missing_data_model,
        Prior(),
        1
    )

    y = extract_vector_parameter(prior_draw, "y")

    theta = extract_vector_parameter(prior_draw, "θ")

    data_model = simple_hierarchical_normal(
        ;
        y = y.value,
        participant_id = participant_id,
        N_participants = N_participants
    )

    chain = sample(Xoshiro(0), data_model, NUTS(), MCMCThreads(), 1000, 4)

    chain, ground_truth, theta

end

let
    f = Figure()

    plot_hyper_parameters_vs_fitted!(
        f,
        chain,
        ground_truth
    )

end

let 
    f = Figure()

    plot_random_effects_fitted_vs_true!(
        f,
        chain,
        theta
    )

end


