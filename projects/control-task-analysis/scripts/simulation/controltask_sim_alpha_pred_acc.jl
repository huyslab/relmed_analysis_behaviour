import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Statistics
using Chain, Tidier, AlgebraOfGraphics, ColorSchemes

CairoMakie.activate!(type="svg")
includet("ControlEffortTask.jl")
using .ControlEffortTask

function actor_fn(stimulus)
    chosen_boat = stimulus.boats[rand(1:2)]
    # chosen_boat = stimulus.boats[1]  # Always choose the first boat for simplicity
    effort = rand(DiscreteUniform(5, 25))
    return (; chosen_boat, effort)
end

# Main execution
if !@isdefined(results_df) || true  # Only run if not already defined
    # Test with different alpha values
    alpha_values = [1.0, 10.0, 100.0]

    @info "Starting alpha simulation study..."
    results_df = run_alpha_simulation(alpha_values;
        n_participants=100, n_particles=100, actor=actor_fn, n_trials=72)

    @info "Creating visualizations..."
    fig = plot_alpha_comparison(results_df)
    display(fig)

    @info "Analyzing final performance..."
    final_perf = analyze_final_performance(results_df)
    println(final_perf)

    # Save results
    # CSV.write("alpha_simulation_results.csv", results_df)
    # @info "Results saved to alpha_simulation_results.csv"
end

# Run comprehensive analysis with both general and specific belief changes
comprehensive_results = comprehensive_belief_change_analysis(
    target_matrix_idx=24,
    alpha_values=[1.0, 10.0, 100.0],
    n_participants=100,
    n_particles=100,
    n_trials=72
);

# Display the comprehensive plot
display(comprehensive_results.fig_patterns)
# Or display the focused target matrix changes plot
display(comprehensive_results.fig_focus)
# Look at the summary focusing on general belief shift
println(comprehensive_results.summary_df)