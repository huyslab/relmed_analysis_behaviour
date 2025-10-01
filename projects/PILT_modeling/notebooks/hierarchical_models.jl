# Simple hierarchical regression models

# Setup environment and imports
begin
    cd("/home/jovyan")
    import Pkg
    # Activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # Instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

    using Turing, DynamicPPL, Distributions, Random, DataFrames
    include("$(pwd())/core/model_utils.jl")

    script_dir = dirname(@__FILE__)
    include(joinpath(script_dir, "..", "utils", "plotting.jl"))
    include(joinpath(script_dir, "..", "utils", "modeling.jl"))
    include(joinpath(script_dir, "..", "config.jl"))

    include(joinpath(script_dir, "..", "models", "hierarchy_devel.jl"))
end

# Parameter recovery study: Normal hierarchical model
fs_bernoulli_logit_normal = let normal_ground_truth_priors = Dict(
        :μ => Dirac(1.),
        :σ_participant => Dirac(0.6),
        :σ => Dirac(0.5)
    )

    # Simulate data and fit normal hierarchical model
    chain, theta = simulate_fit_hierarchical(
        simple_hierarchical_normal;
        ground_truth_priors = normal_ground_truth_priors
    )

    # Plot fixed effects recovery
    f1 = Figure()
    plot_fixed_effects_recovery!(
        f1,
        chain,
        normal_ground_truth_priors
    )

    # Plot random effects recovery
    f2 = Figure()
    plot_random_effects_recovery!(
        f2,
        chain,
        theta
    )

    f1, f2
end

# Parameter recovery study: Bernoulli logit hierarchical model
fs_bernoulli_logit = let bernoulli_ground_truth_priors = Dict(
        :μ => Dirac(1.),
        :σ_participant => Dirac(0.6)
    )

    # Simulate data and fit Bernoulli logit hierarchical model
    chain, theta = simulate_fit_hierarchical(
        simple_hierarchical_bernoulli_logit;
        ground_truth_priors = bernoulli_ground_truth_priors
    )

    # Plot fixed effects recovery
    f1 = Figure()
    plot_fixed_effects_recovery!(
        f1,
        chain,
        bernoulli_ground_truth_priors
    )

    # Plot random effects recovery
    f2 = Figure()
    plot_random_effects_recovery!(
        f2,
        chain,
        theta
    )

    f1, f2
end
