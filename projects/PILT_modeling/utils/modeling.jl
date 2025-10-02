
"""
Auxillary functions for working with Turing.jl models and MCMC chains.
"""

using Turing, DataFrames

"""
    extract_array_parameter(chain::Chains, parameter::String) -> DataFrame

Extract all elements of a vector or matrix parameter from a Turing.jl MCMC chain into long format.

# Arguments
- `chain::Chains`: MCMC chain containing samples
- `parameter::String`: Base parameter name (e.g., "beta" for β[1], β[2], ... or "theta" for θ[1,1], θ[1,2], ...)

# Returns
Long-format DataFrame with columns: iteration, chain, parameter, value
For matrices, parameter names will be in format "param[row,col]"
"""
function extract_array_parameter(
    chain::Chains,
    parameter::String
)   
    # Create regex patterns to match both vector and matrix parameter names
    vector_regex = Regex("^$(parameter)\\[\\d+\\]\$")  # Matches parameter[1], parameter[2], etc.
    matrix_regex = Regex("^$(parameter)\\[\\d+, \\d+\\]\$")  # Matches parameter[1, 1], parameter[1, 2], etc.
    
    # Get all parameter names from the chain
    all_names = string.(names(chain))
    
    # Filter to find vector or matrix elements
    vector_names = filter(name -> occursin(vector_regex, name), all_names)
    matrix_names = filter(name -> occursin(matrix_regex, name), all_names)
    
    # Determine which type we have and use appropriate names
    if !isempty(vector_names) && isempty(matrix_names)
        param_names = vector_names
    elseif isempty(vector_names) && !isempty(matrix_names)
        param_names = matrix_names
    elseif !isempty(vector_names) && !isempty(matrix_names)
        # If both exist, prefer matrix (more specific)
        param_names = matrix_names
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

"""
    simulate_fit_hierarchical(model; N_participants, N_obs, ground_truth_priors, rng)

Simulate data from a hierarchical model and fit it back to assess parameter recovery.

# Arguments
- `model::Function`: Turing model function to simulate from and fit
- `N_participants::Int`: Number of participants (default: 50)
- `N_obs::Int`: Number of observations per participant (default: 200)
- `ground_truth_priors::Dict`: True parameter values as Dirac distributions
- `rng::AbstractRNG`: Random number generator (default: Xoshiro(0))

# Returns
- `chain::Chains`: MCMC samples from fitted model
- `theta::DataFrame`: True random effects in long format
"""
function simulate_fit_hierarchical(
    model::Function;
    N_participants::Int = 50,
    N_obs::Int = 200,
    ground_truth_priors::Dict,
    rng::AbstractRNG = Xoshiro(0)
) 
    # Create participant index vector
    participant_id = repeat(1:N_participants, inner=N_obs)

    # Sample synthetic data from prior predictive distribution
    missing_data_model = model(
        ;
        y = missing,
        participant_id = participant_id,
        N_participants = N_participants,
        priors = ground_truth_priors
    )

    prior_draw = sample(
        rng,
        missing_data_model,
        Prior(),
        1
    )

    # Extract simulated data and true random effects
    y = extract_array_parameter(prior_draw, "y")
    theta = extract_array_parameter(prior_draw, "θ")

    # Fit model to simulated data
    data_model = model(
        ;
        y = y.value,
        participant_id = participant_id,
        N_participants = N_participants
    )

    # Run MCMC sampling
    chain = sample(rng, data_model, NUTS(), MCMCThreads(), 1000, 4)

    chain, theta
end
