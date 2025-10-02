# Simple hierarchical regression models

# Setup environment and imports
begin
    cd("/home/jovyan")
    import Pkg
    # Activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # Instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

    using Turing, DynamicPPL, Distributions, Random, DataFrames, JLD2
    include("$(pwd())/core/model_utils.jl")

    script_dir = dirname(@__FILE__)
    saved_models_dir = joinpath(script_dir, "..", "saved_models")
    include(joinpath(script_dir, "..", "utils", "plotting.jl"))
    include(joinpath(script_dir, "..", "utils", "modeling.jl"))
    include(joinpath(script_dir, "..", "config.jl"))

    include(joinpath(script_dir, "..", "models", "hierarchy_devel.jl"))
end

# Parameter recovery study: Normal hierarchical model
fs_bernoulli_logit_normal = let normal_ground_truth_priors = Dict(
        :μ => Dirac(1.),
        :τ => Dirac(0.6),
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
        :τ => Dirac(0.6)
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

# Parameter recovery study: Logistic regression hierarchical model
# fs_logistic_regression = 
fs_logistic_regression = let model_name = "logistic_regression", N_participants = 50, N_obs = 100, N_predictors = 2, logistic_ground_truth_priors = Dict(
        :β => arraydist([Dirac(1.), Dirac(0.8)]),  # Coefficients for intercept and slope
        :τ => arraydist([Dirac(0.5), Dirac(0.3)]),
        :η => 2.0
    ), rng = Xoshiro(0)

    # Create participant index vector
    participant_id = repeat(1:N_participants, inner=N_obs)

    # Create model matrix
    X = hcat(ones(N_participants * N_obs), [randn(rng, N_participants * N_obs) for _ in 1:(N_predictors-1)]...) # Intercept + N_predictors-1 continuous predictors

    # Sample synthetic data from prior predictive distribution
    missing_data_model = hierarchical_logistic_regression_one_grouping(
        ;
        y = missing,
        X = X,
        N_predictors = N_predictors,
        participant_id = participant_id,
        N_participants = N_participants,
        priors = logistic_ground_truth_priors
    )

    prior_draw = sample(
        rng,
        missing_data_model,
        Prior(),
        1
    )

    # Extract simulated data and true random effects
    y = extract_array_parameter(prior_draw, "y")
    z = extract_array_parameter(prior_draw, "z")

    # Fit model to simulated data
    data_model = hierarchical_logistic_regression_one_grouping(
        ;
        y = y.value,
        X = X,
        N_predictors = N_predictors,
        participant_id = participant_id,
        N_participants = N_participants
    )

    # Run MCMC sampling
    chain = load_or_run(joinpath(saved_models_dir, model_name), () -> sample(rng, data_model, NUTS(), MCMCThreads(), 1000, 4))


    # Plot fixed effects recovery
    f1 = Figure()
    plot_fixed_effects_recovery!(
        f1,
        chain,
        logistic_ground_truth_priors
    )

    # Plot random effects recovery
    f2 = Figure()
    plot_random_effects_recovery!(
        f2[1,1],
        chain,
        filter(r -> startswith(r.parameter, "z[1,"), z)
    )

    plot_random_effects_recovery!(
        f2[1,2],
        chain,
        filter(r -> startswith(r.parameter, "z[2,"), z)
    )


    f1, f2
end
