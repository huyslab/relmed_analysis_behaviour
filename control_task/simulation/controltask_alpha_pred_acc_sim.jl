import Pkg
Pkg.activate("relmed_environment")
Pkg.instantiate()

using Revise, Random, DataFrames, CairoMakie, Distributions, CSV, JSON, Statistics
using Chain, Tidier, AlgebraOfGraphics, ColorSchemes

CairoMakie.activate!(type = "svg")
includet("ControlEffortTask.jl")
using .ControlEffortTask

# Main execution
if !@isdefined(results_df) || true  # Only run if not already defined
    # Test with different alpha values
    alpha_values = [0.025, 0.05, 1.0, 10.0, 100.0, 1000.0]
    
    function actor_fn(stimulus) 
        chosen_boat = stimulus.boats[rand(1:2)]
        effort = rand(DiscreteUniform(1, 30))
        return (; chosen_boat, effort)
    end

    @info "Starting alpha simulation study..."
    results_df = run_alpha_simulation(alpha_values; n_participants=100, actor_fn = actor_fn, n_trials=72)
    
    # @info "Creating visualizations..."
    # fig = plot_alpha_comparison(results_df)
    # display(fig)
    
    @info "Analyzing final performance..."
    final_perf = analyze_final_performance(results_df)
    println(final_perf)
    
    # Save results
    CSV.write("alpha_simulation_results.csv", results_df)
    @info "Results saved to alpha_simulation_results.csv"
end

@chain results_df begin
    @group_by(alpha, trial)
    @summarize(mean_accuracy = mean(accuracy), std_accuracy = std(accuracy))
    @ungroup()
    data(_) * mapping(:trial, :mean_accuracy, group=:alpha => nonnumeric, color=:alpha => nonnumeric) * visual(Lines)
    draw(scales(Color = (; palette = :Oranges_6)); axis =(; title = "Alpha Simulation Results", xlabel = "Trial", ylabel = "Mean Accuracy"))
end