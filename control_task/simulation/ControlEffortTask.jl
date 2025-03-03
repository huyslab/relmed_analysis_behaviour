module ControlEffortTask

using DataFrames, LinearAlgebra, LogExpFunctions, Distributions, StatsBase, Parameters, Combinatorics, Plots, ProgressMeter, CSV, Random, Debugger

using AlgebraOfGraphics, ColorSchemes, CairoMakie

set_theme!(theme_minimal())
colors = ColorSchemes.Paired_10.colors;
inch = 96
pt = 4/3
cm = inch / 2.54

#==========================================================
Types
==========================================================#

@with_kw mutable struct TaskParameters
    n_trials::Int = 144
    n_states::Int = 4
    n_particles::Int = 100
    n_matrices::Int = factorial(n_states)
    n_samples::Int = n_matrices * n_particles
    alpha::Float64 = 10.0
    sigma::Float64 = 1.0
    beta_true::Vector{Float64} = [5.0, 10.0, 20.0]
    Tc::Vector{Int} = [4, 3, 2, 1]
    Tu::Vector{Int} = [2, 3, 4, 1]
end

struct TrialDiagnostics
    effective_sample_size::Float64
    particle_diversity::Float64
    likelihood::Float64
    resampled::Bool
end

struct ExperimentResults
    beta_estimates::Matrix{Float64}
    weight_estimates::Matrix{Float64}
    control_estimates::Matrix{Float64}
    diagnostics::Vector{TrialDiagnostics}
end

#==========================================================
Core Simulation Functions
==========================================================#

function initialize_transition_matrices(pars::TaskParameters)
    @unpack n_states, n_matrices, Tc = pars
    matrices = zeros(n_states, n_states, n_matrices) # r: next state by c: current state by n_matrices
    matrix_index = 1
    true_index = 0
    
    for perms in permutations(1:n_states)
        matrices[:, :, matrix_index] = Float64.(I(n_states)[:, perms])
        if perms == Tc
            true_index = matrix_index
        end
        matrix_index += 1
    end
    
    return matrices, true_index
end

"""
Generates a trial dataset either randomly, from a factorial design, or from a CSV file.


Parameters:
- pars: TaskParameters struct
- mode: :random, :factorial, or :csv
- csv_path: path to CSV file (only used if mode=:csv)
- shuffle: whether to shuffle the trials (default: true)

Usage:
```julia
# Random generation
trials = generate_dataset(pars)  # defaults to random mode
trials = generate_dataset(pars, mode=:random)

# Factorial design
trials = generate_dataset(pars, mode=:factorial)

# Without shuffling
trials = generate_dataset(pars, mode=:factorial, shuffle=false)

# From CSV file
trials = generate_dataset(pars, mode=:csv, csv_path="trials.csv")
```
"""
function generate_dataset(pars::TaskParameters;
    mode::Symbol=:random,
    csv_path::Union{String,Nothing}=nothing,
    shuffle::Bool=true)

    # @unpack n_trials, n_states, beta_true, Tc, Tu = pars

    function generate_trial(current_state, boats, wind)
        # Action model part
        chosen_boat = boats[rand(1:2)]
        effort = rand(DiscreteUniform(0, 35))

        beta = pars.beta_true[wind]
        p_control = 1.0 / (1.0 + exp(-2.0 * (effort - beta)))
        is_controlled = rand() < p_control

        next_state = is_controlled ? pars.Tc[chosen_boat] : pars.Tu[current_state]

        return (; current_state, boats, wind, chosen_boat, effort,
            next_state, controlled=is_controlled, p_control)
    end

    trials = if mode == :random
        [generate_trial(rand(1:4), rand(1:4, 2), rand(1:3)) 
        for _ in 1:pars.n_trials]

    elseif mode == :factorial
        trials = []
        for current_state in 1:pars.n_states
            for wind in 1:3
                for (boat1, boat2) in permutations(1:pars.n_states, 2)
                    push!(trials, generate_trial(current_state, [boat1, boat2], wind))
                end
            end
        end
        trials

    elseif mode == :csv
        isnothing(csv_path) && error("CSV path must be provided when mode=:csv")

        # Read CSV file
        df = CSV.read(csv_path, DataFrame)
        required_cols = ["current_state", "boat1", "boat2", "wind"]

        # Verify required columns exist
        missing_cols = setdiff(required_cols, names(df))
        if !isempty(missing_cols)
            error("Missing required columns in CSV: $(join(missing_cols, ", "))")
        end

        # Generate trials from CSV data
        [generate_trial(row.current_state,
            [row.boat1, row.boat2],
            row.wind)
        for row in eachrow(df)]

    else
        error("Invalid mode: $mode. Must be :random, :factorial, or :csv")
    end

    shuffle && shuffle!(trials)
    return convert(Vector{NamedTuple},trials)
end

# Generate stimulus sequence
function generate_stimuli(pars::TaskParameters;
                         mode::Symbol=:random,
                         csv_path::Union{String,Nothing}=nothing,
                         shuffle::Bool=true)
    
    stimuli = if mode == :random
        [
            (; current_state=rand(1:4), boats=rand(1:4, 2), wind=rand(1:3)) 
            for _ in 1:pars.n_trials
        ]
        
    elseif mode == :factorial
        stimuli_list = []
        for current_state in 1:pars.n_states
            for wind in 1:3
                for (boat1, boat2) in permutations(1:pars.n_states, 2)
                    push!(stimuli_list, (;current_state, boats=[boat1, boat2], wind))
                end
            end
        end
        stimuli_list
        
    elseif mode == :csv
        isnothing(csv_path) && error("CSV path must be provided when mode=:csv")
        
        # Read CSV file
        df = CSV.read(csv_path, DataFrame)
        required_cols = ["current_state", "boat1", "boat2", "wind"]
        
        # Verify required columns exist
        missing_cols = setdiff(required_cols, names(df))
        if !isempty(missing_cols)
            error("Missing required columns in CSV: $(join(missing_cols, ", "))")
        end
        
        # Generate stimuli from CSV data
        [
            (;current_state=row.current_state, boats=[row.boat1, row.boat2], wind=row.wind)
            for row in eachrow(df)
        ]
    else
        error("Invalid mode: $mode. Must be :random, :factorial, or :csv")
    end
    
    shuffle && shuffle!(stimuli)
    return stimuli
end

# Generate action sequence either randomly or based on stimuli
function generate_actions(stimulus::Union{Nothing, NamedTuple}; actor::Function=nothing)
    if isnothing(actor)
        # Default random action generator
        chosen_boat = stimulus.boats[rand(1:2)]
        effort = rand(DiscreteUniform(0, 35))
        return (;chosen_boat, effort)
    else
        # Custom action generator provided as a function
        # This function should return an tuple that contained
        # the chosen boat and effort level
        return actor(stimulus)
    end
end

#==========================================================
Particle Filter Functions
==========================================================#

function compute_trial_diagnostics(particle_weights::Vector{Float64}, likelihood::Vector{Float64}, resampled::Bool)
    ll = sum(log.(likelihood .+ eps()))/length(particle_weights)
    ess = 1.0 / sum(particle_weights .^ 2)
    diversity = -sum(w * log(w + eps()) for w in particle_weights)

    return TrialDiagnostics(ess, diversity, ll, resampled)
end

function update_particles!(betas::Matrix{Float64}, weights::Matrix{Float64}, particle_weights::Vector{Float64}, trial_data::NamedTuple, Tcs::Array{Float64, 3}, pars::TaskParameters)
    @unpack n_samples, sigma, alpha, Tu = pars
    p_controls = zeros(n_samples, 2) # pcx
    p_next_states = zeros(n_samples) # prx

    for i in 1:n_samples
        # Add noise to beta and w(eight) parameters
        betas[trial_data.wind, i] += rand(Normal(0, sigma))
        weights[:, i] = rand(Dirichlet(weights[:, i] * alpha)) .+ 1e-50
        weights[:, i] ./= sum(weights[:, i])

        # Calculate likelihood of new observation
        p_control = 1.0 / (1.0 + exp(-2.0 * (trial_data.effort - betas[trial_data.wind, i])))
        p_next_state_given_control = Tcs[trial_data.next_state, trial_data.chosen_boat, :]' * weights[:, i]
        p_next_state_given_uncont = Float64(Tu[trial_data.current_state] == trial_data.next_state)
        p_next_states[i] = p_next_state_given_control * p_control + p_next_state_given_uncont * (1.0 - p_control)

        particle_weights[i] *= p_next_states[i]

        # Calculate p(c_t|a_t,s_t,w_t,e_t,n_t,beta_t)
        p_controls[i, :] = [p_next_state_given_control * p_control, p_next_state_given_uncont * (1.0 - p_control)]
        p_controls[i, :] ./= sum(p_controls[i, :])
    end
    
    # Normalize weights
    total_weight = sum(particle_weights)
    if total_weight > 0
        particle_weights ./= total_weight
    else
        particle_weights .= 1.0 / n_samples
    end

    return p_controls, p_next_states
end

function resample_particles!(betas::Matrix{Float64}, weights::Matrix{Float64}, particle_weights::Vector{Float64})
    n_samples = length(particle_weights)
    indices = wsample(1:n_samples, particle_weights, n_samples)
    betas .= betas[:, indices]
    weights .= weights[:, indices]
    particle_weights .= 1.0 / n_samples
end

function particle_filter(pars::TaskParameters, dataset::Vector{<:NamedTuple})
    @unpack n_states, n_samples, n_particles, n_trials, beta_true = pars

    # betas, weights, controls, particle_weights are all by n_samples
    particle_weights = ones(n_samples) / n_samples # x; particle weights

    n_betas = length(beta_true)
    betas = randn(n_betas, n_samples) .+ beta_true * 0.1 .+ 10.0 # bx - 0.1 * betatrue'; betatrue' could be the prior mean?; n_samples by n_betas
    
    n_transmats = factorial(n_states)
    Tcs, _ = initialize_transition_matrices(pars)
    weights = zeros(n_transmats, n_samples) # wx

    for i in 1:n_transmats
        alpha = ones(n_transmats)
        alpha[i] = 10
        weights[:, (1:n_particles) .+ (i - 1) * n_particles] = rand(Dirichlet(alpha * 100), n_particles)
    end

    beta_estimates = zeros(n_betas, n_trials) # betahat
    weight_estimates = zeros(n_transmats, n_trials) # wx
    control_estimates = zeros(2, n_trials) # pc; but transposed

    diagnostics = Vector{TrialDiagnostics}(undef, n_trials)

    progress = Progress(n_trials, desc="Running particle filter...")

    for trial in 1:n_trials
        trial_data = dataset[trial]
        p_controls, p_next_states = update_particles!(betas, weights, particle_weights, trial_data, Tcs, pars)

        # Evaluate mean variables
        beta_estimates[:, trial] = betas * particle_weights
        weight_estimates[:, trial] = weights * particle_weights
        control_estimates[:, trial] = p_controls' * particle_weights

        diagnostics[trial] = compute_trial_diagnostics(
            particle_weights, p_next_states, false)

        if diagnostics[trial].effective_sample_size < 0.7 * n_samples # or usually 0.5 * n_samples
            resample_particles!(betas, weights, particle_weights)
            diagnostics[trial] = compute_trial_diagnostics(
                particle_weights, p_next_states, true)
        end

        next!(progress)
    end

    @info "Resampled $(sum([d.resampled for d in diagnostics])) trials"

    return ExperimentResults(beta_estimates, weight_estimates, control_estimates, diagnostics)
end

#==========================================================
Visualization Functions
==========================================================#

function moving_average(x::Vector, window::Int)
    out = zeros(length(x))
    for i in 1:length(x)
        start_idx = max(1, i - window + 1)
        out[i] = mean(x[start_idx:i])
    end
    return out
end

function plot_estimates(results::ExperimentResults, dataset::Vector{<:NamedTuple}, pars::TaskParameters)
    fig = Figure(; figure_padding=0.1cm, size=(16cm, 8cm), fontsize=8pt)

    subfig_r1 = GridLayout(1, 4)
    colsize!(subfig_r1, 1, Fixed(4cm))
    colsize!(subfig_r1, 2, Auto())
    colsize!(subfig_r1, 3, Fixed(4cm))
    colgap!(subfig_r1, 3, 0.1cm)
    colsize!(subfig_r1, 4, Auto())

    subfig_r2 = GridLayout(1, 4)
    colsize!(subfig_r2, 1, Fixed(4cm))
    colsize!(subfig_r2, 2, Fixed(3cm))
    colgap!(subfig_r2, 2, 0cm)
    colsize!(subfig_r2, 3, Auto())
    colsize!(subfig_r2, 4, Fixed(4cm))

    fig.layout[1, 1] = subfig_r1
    fig.layout[2, 1] = subfig_r2

    # Online control estimates
    dataset = DataFrame(dataset)
    dataset.trial = 1:nrow(dataset)
    p_control_df = DataFrame(trial=1:nrow(dataset), p_control=results.control_estimates[1, :])

    p1 = data(dataset) *
         mapping(:trial, :controlled => Int) *
         visual(Scatter, color=(colors[2], 0.9), markersize=4) +
         data(p_control_df) *
         mapping(:trial, :p_control) *
         visual(Lines, color=colors[1], linewidth=0.9)
    draw!(fig[1, 1][1, 1], p1; axis=(; title="Control inference", xlabel="Trial", ylabel="P(Control)"))

    # Control estimate accuracy
    control_acc_df = DataFrame(
        condition=["C=1", "C=0"],
        accuracy=[mean(results.control_estimates[1, dataset.controlled]), mean(results.control_estimates[2, Not(dataset.controlled)])]
    )

    p2 = data(control_acc_df) *
         mapping(:condition => presorted, :accuracy, color=:condition => presorted) *
         visual(BarPlot, color=:condition, width=0.8)
    draw!(fig[1, 1][1, 2], p2; axis=(; title="Control events", xlabel="Condition", ylabel="Accuracy"))

    # Transition matrix weight
    weights_df = DataFrame(results.weight_estimates, :auto)
    transform!(weights_df, eachindex => :x)
    long_weights_df = stack(weights_df, Not(:x))
    transform!(long_weights_df, :variable => ByRow(x -> parse(Int, match(r"[0-9]+", x).match)) => :y)

    p3 = data(long_weights_df) *
         mapping(:y, :x, :value) *
         visual(Heatmap)
    p3_leg = draw!(fig[1, 1][1, 3], p3, scales(Color=(; colormap=ColorSchemes.linear_blue_95_50_c20_n256)); axis=(; title="Transition weights over time", xlabel="Trial", ylabel="Matrix weight"))
    colorbar!(fig[1, 1][1, 4], p3_leg; label="Weight")

    # Betas over time
    betas_df = DataFrame(results.beta_estimates', :auto)
    betas_df.trial = 1:nrow(betas_df)
    long_betas_df = stack(betas_df, Not(:trial))
    beta_labels = ["Low", "Mid", "High"]
    transform!(long_betas_df, :variable => ByRow(x -> beta_labels[parse(Int, match(r"[0-9]+", x).match)]) => :labels)

    p4 = data(long_betas_df) *
         mapping(:trial, :value => "β", color=:labels => presorted => "") *
         visual(Lines, label="Estimated") +
         data((; β=pars.beta_true, labels=beta_labels)) *
         mapping(:β, color=:labels => presorted => "") *
         visual(HLines, linestyle=:dash, label="Ground\ntruth")
    draw!(fig[2, 1][1, 1], p4, scales(Color=(; palette=:Reds_3)); axis=(; title="Betas over time", xlabel="Trial", ylabel="Estimated β"))

    # Effort functions
    effort_df = crossjoin(
        DataFrame(effort=0:30),
        DataFrame(estimated_beta=results.beta_estimates[:, end], true_beta=pars.beta_true, label=beta_labels)
    )
    transform!(effort_df, [:effort, :estimated_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_estimated)
    transform!(effort_df, [:effort, :true_beta] => ByRow((x, y) -> 1 / (1 + exp(-2 * (x - y)))) => :p_control_true)

    p5 = data(effort_df) *
         mapping(:effort, :p_control_true, color=:label => presorted => "") *
         visual(Lines, linestyle=:dash, label="Ground\ntruth") +
         data(effort_df) *
         mapping(:effort, :p_control_estimated, color=:label => presorted => "") *
         visual(Lines, linestyle=:solid, label="Estimated")
    p5_leg = draw!(fig[2, 1][1, 2], p5, scales(Color=(; palette=:Reds_3)); axis=(; title="Effort functions", xlabel="Effort", ylabel="P(Control)"))
    legend!(fig[2, 1][1, 3], p5_leg; tellheight=false, tellwidth=false, padding=(0, 0, 0, 0), gridshalign=:left, patchsize=(10, 10), rowgap=0)

    Box(fig[2, 1][1, 4]; visible=false)

    return fig
end


#==========================================================
Main Execution Functions
==========================================================#

function run_experiment(pars::TaskParameters, dataset::Vector{<:NamedTuple}; show_plots::Bool=true)
    results = particle_filter(pars, dataset)

    if show_plots
        display(plot_estimates(results, dataset, pars))
    end

    return results
end

# Export main functions
export TaskParameters, run_experiment, generate_dataset, particle_filter, plot_estimates

end # module