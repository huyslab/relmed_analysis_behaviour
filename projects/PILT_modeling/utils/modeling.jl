
"""
Auxillary functions for working with Turing.jl models and MCMC chains.
"""

using Turing, DataFrames

"""
    extract_array_parameter(chain::Chains, parameter::String) -> DataFrame

Extract all elements of a vector, matrix, or vector of vectors parameter from a Turing.jl MCMC chain into long format.

# Arguments
- `chain::Chains`: MCMC chain containing samples
- `parameter::String`: Base parameter name (e.g., "beta" for β[1], β[2], ... or "theta" for θ[1,1], θ[1,2], ... or "gamma" for γ[1][1], γ[2][3], ...)

# Returns
Long-format DataFrame with columns: iteration, chain, parameter, value
For matrices, parameter names will be in format "param[row,col]"
For vector of vectors, parameter names will be in format "param[i][j]"
"""
function extract_array_parameter(
    chain::Chains,
    parameter::String
)   
    # Create regex patterns to match vector, matrix, and vector of vectors parameter names
    vector_regex = Regex("^$(parameter)\\[\\d+\\]\$")  # Matches parameter[1], parameter[2], etc.
    matrix_regex = Regex("^$(parameter)\\[\\d+, \\d+\\]\$")  # Matches parameter[1, 1], parameter[1, 2], etc.
    vector_of_vectors_regex = Regex("^$(parameter)\\[\\d+\\]\\[\\d+\\]\$")  # Matches parameter[1][1], parameter[2][3], etc.
    
    # Get all parameter names from the chain
    all_names = string.(names(chain))
    
    # Filter to find vector, matrix, or vector of vectors elements
    vector_names = filter(name -> occursin(vector_regex, name), all_names)
    matrix_names = filter(name -> occursin(matrix_regex, name), all_names)
    vector_of_vectors_names = filter(name -> occursin(vector_of_vectors_regex, name), all_names)
    
    # Determine which type we have and use appropriate names (prefer most specific format)
    if !isempty(vector_of_vectors_names)
        param_names = vector_of_vectors_names
    elseif !isempty(matrix_names)
        param_names = matrix_names
    elseif !isempty(vector_names)
        param_names = vector_names
    else
        # No matching parameters found
        return DataFrame(iteration=Int[], chain=Int[], parameter=String[], value=Float64[])
    end
    
    # Extract the relevant columns from the chain and convert to DataFrame
    df = DataFrame(chain[:, Symbol.(param_names), :])
    
    # Reshape from wide to long format for easier analysis
    # This transforms columns like β[1], β[2], β[3] or θ[1,1], θ[1,2] into rows with a 'parameter' column
    return stack(df, Not([:iteration, :chain]); variable_name = :parameter)
end

"""
    load_or_run(filename::String, func::Function; force_run::Bool = false)

Load cached results from JLD2 file or run function and cache output.
"""
function load_or_run(
    filename::String,
    func::Function;
    force_run::Bool = false
)
    if isfile(filename  * ".jld2") && !force_run
        println("Loading from $filename")
        return JLD2.load(filename * ".jld2", "result")
    else
        println("Running function and saving to $filename")
        result = func()
        JLD2.save(filename * ".jld2", "result", result)
        return result
    end
end

