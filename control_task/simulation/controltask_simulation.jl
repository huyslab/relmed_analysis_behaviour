import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
@info "Running with $(Threads.nthreads()) threads"

using Revise, Random
includet("ControlEffortTask.jl")
using .ControlEffortTask

pars = TaskParameters(
     n_trials=144,
     beta_true=[2, 5, 8] .* 3,
)

# Set random seed for reproducibility
Random.seed!(123)

# Run single experiment
# dataset = generate_dataset(pars, mode=:factorial, shuffle=true)
# results = run_experiment(pars, dataset; show_plots=true);

# Run batch experiments
stimuli_sequence = generate_stimuli(pars; mode=:factorial, shuffle=true)
results_list, datasets_list, batch_fig = run_batch_experiment(pars, stimuli_sequence; n_participants=100, show_plots=true);