# Script to generate all figures and combine into markdown file
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, AlgebraOfGraphics
    include("$(pwd())/core/experiment-registry.jl")
    include("$(pwd())/core/preprocess_data.jl")
end

# Load and preprocess data
(; PILT, PILT_test, WM, WM_test, reversal, delay_discounting, max_press) = preprocess_project(TRIAL1)
