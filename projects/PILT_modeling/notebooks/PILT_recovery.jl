begin
    cd("/home/jovyan")

    using Turing, DynamicPPL, Distributions, Random, DataFrames, JLD2
    include("$(pwd())/core/model_utils.jl")

    script_dir = dirname(@__FILE__)
    saved_models_dir = joinpath(script_dir, "..", "saved_models")
    include(joinpath(script_dir, "..", "utils", "plotting.jl"))
    include(joinpath(script_dir, "..", "utils", "modeling.jl"))
    include(joinpath(script_dir, "..", "utils", "PILT.jl"))
    include(joinpath(script_dir, "..", "config.jl"))

    include(joinpath(script_dir, "..", "models", "PILT.jl"))
end

task_sequence = load_PILT_sequence(
    joinpath(
        dirname(@__FILE__),
        "..",
        "data",
        "trial1_wk0_sequences.js"
    )
)

function prepare_task_sequences(task_sequence, N_participants)
    task_sequences = crossjoin(
        DataFrame(participant = 1:N_participants),
        task_sequence
    )
    block_starts, block_ends, participant_per_block = _block_ranges(
        task_sequences.block, 
        task_sequences.trial, 
        task_sequences.participant
    )
    return task_sequences, block_starts, block_ends, participant_per_block
end

fs_running_average = let running_average_ground_truth_priors = Dict(
        :logρ => Dirac(1.3),
        :τ => Dirac(0.5)
    ), N_participants = 10, rng = Xoshiro(0),
    model_name = "running_average"

    task_sequences, _, _, _ = prepare_task_sequences(task_sequence, N_participants)

    missing_data_model = hierarchical_running_average(;
        block = task_sequences.block,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = fill(missing, nrow(task_sequences)),
        participant = task_sequences.participant,
        N_participants = N_participants,
        initial_value = 0.0, # Initial Q values,
        priors = running_average_ground_truth_priors
    )

    prior_draw = sample(
        rng,
        missing_data_model,
        Prior(),
        1
    )

    # Extract simulated data and true random effects
    choice = extract_array_parameter(prior_draw, "choice")
    θ = extract_array_parameter(prior_draw, "θ")

    # Fit model to simulated data
    data_model = hierarchical_running_average(
        block = task_sequences.block,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = Int.(choice.value),
        participant = task_sequences.participant,
        N_participants = N_participants,
        initial_value = 0.0, # Initial Q values
    )

    # Run MCMC sampling
    chain = load_or_run(joinpath(saved_models_dir, model_name), () -> sample(rng, data_model, NUTS(), MCMCThreads(), 1000, 4))

    @info "Model fit in: $(MCMCChains.wall_duration(chain)) seconds"

    # Plot fixed effects recovery
    f = Figure(size = (1200, 600))
    plot_fixed_effects_recovery!(
        f[1,1],
        chain,
        running_average_ground_truth_priors
    )

    # Plot random effects recovery
    plot_random_effects_recovery!(
        f[1,2],
        chain,
        θ
    )
    
    f

end

fs_running_average_blockloop = let running_average_ground_truth_priors = Dict(
        :logρ => Dirac(1.3),
        :τ => Dirac(0.5)
    ), N_participants = 200, rng = Xoshiro(0),
    model_name = "running_average_blockloop"

    task_sequences, block_starts, block_ends, participant_per_block = prepare_task_sequences(task_sequence, N_participants)

    missing_data_model = hierarchical_running_average_blockloop(;
        block_starts = block_starts,
        block_ends = block_ends,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = fill(missing, nrow(task_sequences)),
        participant_per_block = participant_per_block,
        N_participants = N_participants,
        initial_value = 0.0, # Initial Q values,
        priors = running_average_ground_truth_priors
    )

    prior_draw = sample(
        rng,
        missing_data_model,
        Prior(),
        1
    )

    # Extract true random effects
    θ = extract_array_parameter(prior_draw, "θ")
    choice = extract_array_parameter(prior_draw, "choice")


    # Fit model to simulated data
    data_model = hierarchical_running_average_blockloop(
        block_starts = block_starts,
        block_ends = block_ends,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = Int.(choice.value),
        participant_per_block = participant_per_block,
        N_participants = N_participants,
        initial_value = 0.0
    )

    init_params = [
        rand(rng, data_model) for _ in 1:4
    ]
    
    chain = load_or_run(joinpath(saved_models_dir, model_name), () -> sample(rng, data_model, NUTS(), MCMCThreads(), 1000, 4; initial_params=init_params))

    @info "Model fit in: $(MCMCChains.wall_duration(chain)) seconds"

    # Plot fixed effects recovery
    # Plot fixed effects recovery
    f = Figure(size = (1200, 600))
    plot_fixed_effects_recovery!(
        f[1,1],
        chain,
        running_average_ground_truth_priors
    )

    # Plot random effects recovery
    plot_random_effects_recovery!(
        f[1,2],
        chain,
        θ
    )
    
    f

end

fs_running_average_parallel = let running_average_ground_truth_priors = Dict(
        :logρ => Dirac(1.3),
        :τ => Dirac(0.5)
    ), N_participants = 200, rng = Xoshiro(0),
    model_name = "running_average_parallel"

    task_sequences, block_starts, block_ends, participant_per_block = prepare_task_sequences(task_sequence, N_participants)

    missing_data_model = hierarchical_running_average_parallel(;
        block_starts = block_starts,
        block_ends = block_ends,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = fill(missing, nrow(task_sequences)),
        participant_per_block = participant_per_block,
        N_participants = N_participants,
        initial_value = 0.0, # Initial Q values,
        priors = running_average_ground_truth_priors
    )

    prior_draw = sample(
        rng,
        missing_data_model,
        Prior(),
        1
    )

    # Extract true random effects
    θ = extract_array_parameter(prior_draw, "θ")
    choice = extract_array_parameter(prior_draw, "choice")


    # Fit model to simulated data
    data_model = hierarchical_running_average_parallel(
        block_starts = block_starts,
        block_ends = block_ends,
        trial = task_sequences.trial,
        outcomes = hcat(task_sequences.feedback_left, task_sequences.feedback_right),
        N_actions = 2,
        choice = Int.(choice.value),
        participant_per_block = participant_per_block,
        N_participants = N_participants,
        initial_value = 0.0
    )

    init_params = [
        rand(rng, data_model) for _ in 1:4
    ]


    chain = load_or_run(joinpath(saved_models_dir, model_name), () -> sample(rng, data_model, NUTS(), MCMCThreads(), 1000, 4; initial_params=init_params); force_run = true)

    @info "Model fit in: $(MCMCChains.wall_duration(chain)) seconds"

    # Plot fixed effects recovery
    # Plot fixed effects recovery
    f = Figure(size = (1200, 600))
    plot_fixed_effects_recovery!(
        f[1,1],
        chain,
        running_average_ground_truth_priors
    )

    # Plot random effects recovery
    plot_random_effects_recovery!(
        f[1,2],
        chain,
        θ
    )
    
    f

end


