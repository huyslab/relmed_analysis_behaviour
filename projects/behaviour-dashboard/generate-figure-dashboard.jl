# Script to generate all figures and combine into markdown file

# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, CairoMakie, Dates

    # Include data scripts
    include("$(pwd())/core/experiment-registry.jl")
    include("$(pwd())/core/preprocess_data.jl")

    script_dir = dirname(@__FILE__)

    # Load configurations and theme
    include(joinpath(script_dir, "config.jl"))

    # Markdown generation
    include(joinpath(script_dir, "utils/markdown.jl"))

    # Include task-specific scripts
    task_dir = joinpath(script_dir, "task-scripts")
    include(joinpath(task_dir, "generate-figures-card-choosing.jl"))
    include(joinpath(task_dir, "generate-figures-reversal.jl"))
    include(joinpath(task_dir, "generate-figures-delay-discounting.jl"))

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results")
    isdir(result_dir) || mkpath(result_dir)
end

# Figure registry for markdown generation
figure_registry = Vector{NamedTuple{(:filename, :title), Tuple{String, String}}}()


# Register figure function
function register_save_figure(filename::String, f::Figure, title::String)
    push!(figure_registry, (filename = filename, title = title))
    save(joinpath(result_dir, filename * ".svg"), f)
    return f
end

# Load and preprocess data
begin 
    (; PILT, PILT_test, WM, WM_test, reversal, delay_discounting, max_press) = preprocess_project(TRIAL1)
end

# Generate PILT learning curve by session
let PILT_main_sessions = filter(x -> x.session != "screening", PILT)
    
    f = Figure(size = (800, 600))
    plot_learning_curves_by_factor!(f, PILT_main_sessions; factor = :session)

    filename = "PILT_learning_curves_by_session"
    register_save_figure(filename, f, "PILT Learning Curves by Session")
end

# Generate WM learning curve by session
let WM_main_sessions = filter(x -> x.session != "screening", WM) |> prepare_WM_data;

    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_factor!(
        f1,
        WM_main_sessions;
        factor = :session,
        xcol = :appearance,
        early_stopping_at = nothing)

    filename1 = "WM_learning_curves_by_session"
    register_save_figure(filename1, f1, "Working Memory Learning Curves by Session")

    f2 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f2, WM_main_sessions; facet = :session, variability = :individuals)
    filename2 = "WM_learning_curves_by_delay_bins_and_session_individuals"
    register_save_figure(filename2, f2, "Working Memory Learning Curves by Delay Bins and Session (Individual Participants)")

    f3 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f3, WM_main_sessions; facet = :session)
    filename3 = "WM_learning_curves_by_delay_bins_and_session_group"
    register_save_figure(filename3, f3, "Working Memory Learning Curves by Delay Bins and Session (Group Average)")

end

# Generate reversal accuracy curve
let preproc_df = preprocess_reversal_data(reversal)

    f = Figure(size = (800, 600))

    plot_reversal_accuracy_curve_by_factor!(f, preproc_df)

    filename = "reversal_accuracy_curve"
    register_save_figure(filename, f, "Reversal Learning Accuracy Curve")
end

# Generate delay discounting curves
let preproc_df = preprocess_delay_discounting_data(delay_discounting)

    describe(preproc_df)
end

# Generate the dashboard
generate_markdown_dashboard()