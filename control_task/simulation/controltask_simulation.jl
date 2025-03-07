import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()
@info "Running with $(Threads.nthreads()) threads"

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON
CairoMakie.activate!(type = "svg")
includet("ControlEffortTask.jl")
using .ControlEffortTask

pars = TaskParameters(
     n_trials=144,
     beta_true=[2, 4, 6] .* 3,
     n_particles=40
)

# Set random seed for reproducibility
Random.seed!(123)

# Run single experiment
# dataset = generate_dataset(pars, mode=:factorial, shuffle=true)
# results = run_experiment(pars, dataset; show_plots=true);

# Run batch experiments
stimuli_sequence = generate_stimuli(pars; mode=:factorial, shuffle=true)

# results_list, datasets_list, batch_fig = run_batch_experiment(pars, stimuli_sequence; n_participants=100, show_plots=true);

# Weaker actors
function actor_fn(stimulus) 
     chosen_boat = stimulus.boats[rand(1:2)]
     effort = rand(DiscreteUniform(3, 24))
     return (; chosen_boat, effort)
end
results_list, datasets_list, batch_fig = run_batch_experiment(pars, stimuli_sequence; n_participants=50, actor = actor_fn, show_plots=true);
display(batch_fig)