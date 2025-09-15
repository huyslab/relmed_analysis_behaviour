
# tests/runtests.jl
using Pkg
Pkg.activate("$(pwd())/environment")
Pkg.instantiate()

include("test_model_utils.jl")