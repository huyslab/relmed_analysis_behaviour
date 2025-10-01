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
    include(joinpath(script_dir, "..", "utils", "modeling.jl"))
    include(joinpath(script_dir, "..", "config.jl"))

    include(joinpath(script_dir, "..", "models", "hierarchy_devel.jl"))
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

    plot_fixed_effects_recovery!(
        f,
        chain,
        ground_truth
    )

end

let 
    f = Figure()

    plot_random_effects_recovery!(
        f,
        chain,
        theta
    )

end


