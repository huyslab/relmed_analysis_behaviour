using Turing, DataFrames

function extract_vector_parameter(
    chain::Chains,
    parameter::String
)   
    regex = Regex("^$(parameter)\\[\\d+\\]\$")
    param_names = filter(name -> occursin(regex, name), string.(names(chain)))
    df = DataFrame(chain[:, Symbol.(param_names), :])
    return stack(df, Not([:iteration, :chain]); variable_name = :parameter)
end
