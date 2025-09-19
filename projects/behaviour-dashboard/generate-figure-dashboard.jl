# Script to generate all figures and combine into markdown file

# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, CairoMakie

    # Include data scripts
    include("$(pwd())/core/experiment-registry.jl")
    include("$(pwd())/core/preprocess_data.jl")

    script_dir = dirname(@__FILE__)

    # Load configurations and theme
    include(joinpath(script_dir, "config.jl"))

    # Include task-specific scripts
    include(joinpath(script_dir, "generate-figures-card-choosing.jl"))

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results")
    isdir(result_dir) || mkpath(result_dir)
end

# Save figure function
function save_fig(filename::String, f::Figure)
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

    save_fig("PILT_learning_curves_by_session", f)
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

    save_fig("WM_learning_curves_by_session", f1)

    f2 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f2, WM_main_sessions; facet = :session, variability = :individuals)
    save_fig("WM_learning_curves_by_delay_bins_and_session_individuals", f2)

    f3 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f3, WM_main_sessions; facet = :session)
    save_fig("WM_learning_curves_by_delay_bins_and_session_group", f3)

end

