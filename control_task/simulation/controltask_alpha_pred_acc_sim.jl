import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Statistics
using Chain, Tidier, AlgebraOfGraphics, ColorSchemes

CairoMakie.activate!(type = "svg")
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
        n_participants=100, n_particles=1000, actor=actor_fn, n_trials=72)
    
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

# Run the comprehensive natural evidence analysis
natural_results = comprehensive_natural_evidence_analysis(
    alpha_values=[1.0, 10.0, 100.0],
    n_particles=1000,
    n_participants=50,
    n_trials=72,
    actor=actor_fn
)

# Display the plots
display(natural_results.fig_patterns)
display(natural_results.fig_timeline)

# Look at the summary
println(natural_results.summary_df)