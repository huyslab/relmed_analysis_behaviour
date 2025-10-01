
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
    y = extract_vector_parameter(prior_draw, "y")
    theta = extract_vector_parameter(prior_draw, "θ")

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
