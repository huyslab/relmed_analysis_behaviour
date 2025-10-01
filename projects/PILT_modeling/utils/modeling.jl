
"""
Auxillary functions for working with Turing.jl models and MCMC chains.
"""

using Turing, DataFrames

"""
    extract_vector_parameter(chain::Chains, parameter::String) -> DataFrame

Extract all elements of a vector parameter from a Turing.jl MCMC chain into long format.

# Arguments
- `chain::Chains`: MCMC chain containing samples
- `parameter::String`: Base parameter name (e.g., "beta" for β[1], β[2], ...)

# Returns
Long-format DataFrame with columns: iteration, chain, parameter, value
"""
function extract_vector_parameter(
    chain::Chains,
    parameter::String
)   
    # Create regex pattern to match vector parameter names like "parameter[1]", "parameter[2]", etc.
    regex = Regex("^$(parameter)\\[\\d+\\]\$")
    
    # Filter chain parameter names to find all elements of the specified vector parameter
    param_names = filter(name -> occursin(regex, name), string.(names(chain)))
    
    # Extract the relevant columns from the chain and convert to DataFrame
    df = DataFrame(chain[:, Symbol.(param_names), :])
    
    # Reshape from wide to long format for easier analysis
    # This transforms columns like β[1], β[2], β[3] into rows with a 'parameter' column
    return stack(df, Not([:iteration, :chain]); variable_name = :parameter)
end
