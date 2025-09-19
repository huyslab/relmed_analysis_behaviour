# Script to generate all figures and combine into markdown file

# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, AlgebraOfGraphics

    # Include data scripts
    include("$(pwd())/core/experiment-registry.jl")
    include("$(pwd())/core/preprocess_data.jl")

    script_dir = dirname(@__FILE__)

    # Load configurations and theme
    include(joinpath(script_dir, "config.jl"))

    # Include task-specific scripts
    include(joinpath(script_dir, "generate-figures-PILT.jl"))

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results")
    isdir(result_dir) || mkpath(result_dir)
end

# Load and preprocess data
begin 
    (; PILT, PILT_test, WM, WM_test, reversal, delay_discounting, max_press) = preprocess_project(TRIAL1)
end

# Generate PILT figures
let PILT_main_sessions = filter(x -> x.session != "screening", PILT)
    
    f = Figure(size = (800, 600))
    plot_learning_curves_by_factor!(f, PILT_main_sessions; factor = :session)

    # save(joinpath(result_dir, "PILT_learning_curve.png"), f)
end
