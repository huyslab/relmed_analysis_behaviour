# This file contains general functions to work with models

## Sample from prior ----------------------------------------------------------------------------
"""
    prior_sample(task::NamedTuple; model::Function, n::Int64=1, priors::Dict, outcome_name::Symbol, rng::AbstractRNG, kwargs...)

Generate samples from the prior distribution of a model using NamedTuple task data.

# Arguments
- `task::NamedTuple`: All needed task variables including outcome measure as array of missing values
- `model::Function`: The model function to sample from
- `n::Int64`: Number of samples to generate (default: 1)
- `priors::Dict`: Prior distributions for model parameters
- `outcome_name::Symbol`: Name of the outcome variable
- `rng::AbstractRNG`: Random number generator
- `kwargs...`: Additional keyword arguments passed to model

# Returns
- Array or vector of prior samples for the outcome variable
"""
function prior_sample(
	task::NamedTuple; # Should contain the outocme measure as an array of missing values
	model::Function,
	n::Int64 = 1, # Number of samples to generate
    priors::Dict,
	outcome_name::Symbol,
	rng::AbstractRNG = Random.default_rng(),
	kwargs... # Key-word arguments to model
)	
	# Pass data to model
    task_model = model(;
        task...,
        priors = priors,
		kwargs... # Key-word arguments to model
    )

	# Draw parameters and simulate choice
	prior_sample = sample(
		rng,
		task_model,
		Prior(),
		n
	)

	# Exctract result to array
	result = prior_sample[:, [Symbol("$outcome_name[$i]") for i in 1:length(task[outcome_name])], 1] |>
		Array |> transpose |> collect

	# Flatter to vector if possible
	if n == 1
		result = vec(result)
	end
	
	return result
end

"""
    prior_sample(task::AbstractDataFrame; model::Function, unpack_function::Function, n::Int64=1, priors::Dict, outcome_name::Symbol, rng::AbstractRNG, kwargs...)

Generate samples from the prior distribution of a model using DataFrame task data that needs to be unpacked.

# Arguments
- `task::AbstractDataFrame`: Task data that will be unpacked using `unpack_function`
- `model::Function`: The model function to sample from
- `unpack_function::Function`: Function to convert DataFrame to NamedTuple
- `n::Int64`: Number of samples to generate (default: 1)
- `priors::Dict`: Prior distributions for model parameters
- `outcome_name::Symbol`: Name of the outcome variable
- `rng::AbstractRNG`: Random number generator
- `kwargs...`: Additional keyword arguments passed to model

# Returns
- Array or vector of prior samples for the outcome variable
"""
function prior_sample(
	task::AbstractDataFrame;
	model::Function,
	unpack_function::Function,
	n::Int64 = 1, # Number of samples to generate
    priors::Dict,
	outcome_name::Symbol,
	rng::AbstractRNG = Random.default_rng(),
	kwargs... # Key-word arguments to model
)

	# Add placeholder for outcome
	task_w_outcome = copy(task)
	task_w_outcome[!, outcome_name] .= missing

	# Simulate
	return prior_sample(
		unpack_function(task_w_outcome);
		model,
		n,
		priors,
		outcome_name,
		rng,
		kwargs... 
	)
end

## MLE / MAP estimation -------------------------------------------------------------------------
"""
    multistart_mode_estimator(model::DynamicPPL.Model; estimator::Union{MLE, MAP}, n_starts::Int64=5) -> OptimizationResult

Estimates the mode of a Turing model using a multistart approach to avoid local optima.

# Arguments
- `model::DynamicPPL.Model`: A Turing model to be optimized
- `estimator::Union{MLE, MAP}`: The mode estimation method (MLE or MAP)
- `n_starts::Int64`: Number of optimization starts (default: 5)

# Returns
- OptimizationResult with the highest log-probability among all starts
"""
function multistart_mode_estimator(
	model::DynamicPPL.Model;
	estimator::Union{MLE, MAP},
	n_starts::Int64 = 5
)
	# Store for results
	best_lp = -Inf
	best_result = nothing

	for i in 1:n_starts

		# Optimize
		fit = Turing.Optimisation.estimate_mode(
			model,
			estimator
		)

		if fit.lp > best_lp
			best_lp = fit.lp
			best_result = fit
		end
	end

	return best_result
end

"""
    optimize(data::NamedTuple; model::Function, estimate::String="MAP", priors::Dict, n_starts::Int64=5, kwargs...)

Fits a probabilistic model to NamedTuple data using mode estimation.

# Arguments
- `data::NamedTuple`: Data containing variables needed by the model
- `model::Function`: The model function to optimize
- `estimate::String`: Estimation method ("MLE" or "MAP", default: "MAP")
- `priors::Dict`: Prior distributions for model parameters
- `n_starts::Int64`: Number of optimization starts (default: 5)
- `kwargs...`: Additional keyword arguments passed to model

# Returns
- OptimizationResult with the highest log-probability
"""
function optimize(
	data::NamedTuple;
	model::Function,
	estimate::String = "MAP",
    priors::Dict,
	n_starts::Int64 = 5,
	kwargs... # Key-word arguments to model
)
	
	# Pass data to model
    data_model = model(;
        data...,
        priors = priors,
		kwargs... # Key-word arguments to model
    )

	if estimate == "MLE"
		fit = multistart_mode_estimator(data_model; estimator = MLE(), n_starts = n_starts)
	elseif estimate == "MAP"
		fit = multistart_mode_estimator(data_model; estimator = MAP(), n_starts = n_starts)
	end

	return fit
end

"""
    optimize_multiple(data::AbstractDataFrame; model::Function, unpack_function::Function, initial=nothing, estimate::String="MAP", priors::Dict, grouping_col::Symbol=:PID, n_starts::Int64=5) -> DataFrame

Optimizes a model for each group in a DataFrame using multistart mode estimation.

# Arguments
- `data::AbstractDataFrame`: DataFrame containing data that will be unpacked per group
- `model::Function`: The model function to optimize
- `unpack_function::Function`: Function to convert DataFrame to NamedTuple
- `initial::Union{Float64, Nothing}`: Initial parameter value (default: nothing)
- `estimate::String`: Estimation method ("MLE" or "MAP", default: "MAP")
- `priors::Dict`: Prior distributions for model parameters
- `grouping_col::Symbol`: Column to group data by (default: :PID)
- `n_starts::Int64`: Number of optimization starts (default: 5)

# Returns
- DataFrame with parameter estimates and log-probabilities for each group
"""
# Find MLE / MAP multiple times
function optimize_multiple(
	data::AbstractDataFrame;
	model::Function,
	unpack_function::Function,
    initial::Union{Float64, Nothing} = nothing,
    estimate::String = "MAP",
    priors::Dict,
	grouping_col::Symbol = :PID,
	n_starts::Int64 = 5
) ::DataFrame
	
	# This will hold parameter estimates
	ests = []

	# Lock for threading
	lk = ReentrantLock()

	# Loop over groupings
	Threads.@threads for p in unique(data[!, grouping_col])

		# Select data
		gdf = filter(x -> x[grouping_col] == p, data)

		# Optimize
		est = optimize(
			unpack_function(gdf);
            model = model,
			estimate = estimate,
            priors = priors,
			n_starts = n_starts
		)

		# Unpack results
		est_tuple = NamedTuple{(grouping_col,)}((gdf[1, grouping_col],))
		est_tuple = merge(est_tuple, (; (p => est.values[p] for p in keys(priors))...))
		est_tuple = merge(est_tuple, (; :lp => est.lp))
        
		# Push
		lock(lk) do
			push!(ests, est_tuple)
		end
	end

	# Return as DataFrame
	return sort(DataFrame(ests), grouping_col)
end

"""
    optimize_multiple_by_factor(df::DataFrame; model::Function, factor::Union{Symbol, Vector{Symbol}}, priors::Dict, unpack_function::Function, remap_columns::Dict) -> DataFrame

Estimate the posterior mode or MLE for a dataset, after splitting by one or more factors.

# Arguments
- `df::DataFrame`: The dataset to be analyzed (DataFrame that needs unpacking)
- `model::Function`: The model function to fit to each subset of the data
- `factor::Union{Symbol, Vector{Symbol}}`: Column(s) to split the data by
- `priors::Dict`: Prior distributions for model parameters
- `unpack_function::Function`: Function to prepare the data for the model
- `remap_columns::Dict`: Dictionary to map column names for the unpack function

# Returns
- DataFrame containing fit results for each factor level combination

# Details
The function fits a specified model to each subset of data grouped by levels of the 
specified factor(s), allowing for tailored analyses across groups within the data.
"""
function optimize_multiple_by_factor(
	df::AbstractDataFrame;
	model::Function,
	factor::Union{Symbol, Vector{Symbol}},
	priors::Dict,
	unpack_function::Function,
	remap_columns::Dict
)

	fits = []

	# For backwards comaptibility
	if isa(factor, Symbol)
		factor = [factor]
	end

	# Levels to run over
	levels = unique(df[!, factor])

	for l in eachrow(levels)

		# Select data
		levels_dict = Dict(col => l[col] for col in names(levels))

		gdf = filter(x -> all(x[col] == levels_dict[col] 
			for col in keys(levels_dict)), df)

		# Subset data
		gdf = filter(x -> x[factor] == l, df)

		# Fit data
		fit = optimize_multiple(
			gdf;
			model = model,
			unpack_function = df -> unpack_function(df; columns = remap_columns),
		    priors = priors,
			grouping_col = :prolific_pid,
			n_starts = 10
		)

		# Add factor variables
		factor_pairs = [col => gdf[!, col][1] for col in factor]

		insertcols!(fit, 1, factor_pairs...)

		push!(fits, fit)
	end

	# Combine to single DataFrame
	fits = vcat(fits...)

	# Sort
	sort!(fits, vcat(factor, [:prolific_pid]))

	return fits

end


## Fisher Information functions -----------------------------------------------------------------

"""
    FI(model::DynamicPPL.Model, params::NamedTuple; summary_method::Function = tr)

Compute the Fisher Information Matrix for a Turing model at given parameter values.

# Arguments
- `model::DynamicPPL.Model`: The Turing model with data included
- `params::NamedTuple`: Parameter names and their current values
- `summary_method::Function`: Function to summarize the FIM (default: tr for trace)

# Returns
- Summary of the Fisher Information Matrix (default: trace)

# Details
Computes the negative Hessian of the log-likelihood function with respect to parameters.
"""

"""
    FI(model::DynamicPPL.Model, params::NamedTuple; summary_method::Function = tr)

Compute the Fisher Information Matrix for a Turing model and dataset at given parameter values.

# Arguments
- `model::DynamicPPL.Model`: The Turing model for which to compute the Fisher Information Matrix. Data should be provided in the model.
- `params::NamedTuple`: A named tuple containing parameter names and their current values.
- `summary_method::Function`: An optional function to summarize the Fisher Information Matrix. Defaults to `tr` (trace of the matrix). Possible alternate value: `det` (matrix determinant).

# Returns
- Summary of the Fisher Information Matrix. By default, this is the trace of the matrix.

# Details
The Fisher Information Matrix is computed as the negative Hessian of the log-likelihood function with respect to the parameters. The log-likelihood function is evaluated using the Turing model and the parameter values provided.
"""
function FI(
	model::DynamicPPL.Model,
	params::NamedTuple;
	summary_method::Function = tr
)

	# Exctract param names and value from NamedTuple
	param_names = keys(params)
	param_values = collect(values(params))

	# Define loglikelihood function for ForwardDiff, converting vector of parameter value needed by ForwardDiff.hessian to NamedTuple needed by Turing model
	ll(x) = loglikelihood(model, (;zip(param_names, x)...))

	# Compute FI as negative hessian
	FI = -ForwardDiff.hessian(ll, param_values)

	# Return trace
	return summary_method(FI)
end

"""
    FI(; data::DataFrame, model::Function, map_data_to_model::Function, param_names::Vector{Symbol}, id_col::Symbol=:PID, summary_method::Function=tr, kwargs...)

Compute the Fisher Information for multiple simulated datasets at the true parameter values used to generate the data.

# Arguments
- `data::DataFrame`: DataFrame containing the simulated datasets (grouped by `id_col`)
- `model::Function`: A Turing model
- `map_data_to_model::Function`: Function that maps DataFrame to NamedTuple for model
- `param_names::Vector{Symbol}`: Names of parameters for Fisher Information computation
- `id_col::Symbol`: Column name to split dataset into groups (default: `:PID`)
- `summary_method::Function`: Function to summarize the FIM (default: tr for trace)
- `kwargs...`: Additional keyword arguments passed to model

# Returns
- Sum of Fisher Information computed for each group in the dataset
"""
function FI(;
	data::DataFrame,
	model::Function,
	map_data_to_model::Function, # Function with data::AbstractDataFrame argument returing NamedTuple to pass to model
	param_names::Vector{Symbol},
	id_col::Symbol = :PID, # Column to split the dataset on,
	summary_method::Function = tr,
	kwargs... # Key-word arguments to model
)

	res = 0
	for g in unique(data[!, id_col])

		# Select data
		gdf = filter(x -> x[id_col] == g, data)
		
		# Pass data to model
		m = model(;
			map_data_to_model(
				gdf
			)...,
			kwargs...
		)

		# Extract params from simulated data
		params = (; zip(param_names, collect(gdf[1, param_names]))...)

		# Compute Fisher Information
		res += FI(m, params; summary_method = summary_method)
	end

	return res

end


## Posterior Predictive Sampling -----------------------------------------------------------------
"""
    posterior_predictive(
        data::NamedTuple;
        model::Function,
        fit,
        outcome_name::Symbol,
        n::Int64 = 1,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...
    ) -> NamedTuple

Posterior predictive simulation using point-estimated parameters.

Builds Dirac priors at the supplied parameter values (from `fit` or a mapping),
simulates `outcome_name` via `Prior()` sampling, and computes the log-likelihood
of the observed outcomes under those parameters.

# Arguments
- `data::NamedTuple`: Data mapped for the Turing model (e.g., from an unpack function). If `outcome_name` is present, it will be replaced with `missing` values for simulation/prediction.
- `model::Function`: Turing model function.
- `fit`: Either a Turing optimization result with `.values`, or a `NamedTuple`/`Dict`
  of parameter values keyed by parameter `Symbol`s.
- `outcome_name::Symbol`: The dependent variable name to simulate (e.g., `:choice`).
- `n::Int64`: Number of posterior predictive draws (replicates). Defaults to 1.
- `rng::AbstractRNG`: Random number generator.
- `kwargs...`: Any extra keyword args passed to `model` (e.g., `initV`).

# Returns
- `(; predicted, loglike)`: `predicted` is a vector (if `n==1`) or a matrix of simulated
  outcomes (N x n). `loglike` is the log-likelihood of the observed outcomes under the
  provided parameter values (or `missing` if not computable).
"""
function posterior_predictive(
    data::NamedTuple;
    model::Function,
    fit,
    outcome_name::Symbol,
    n::Int64 = 1,
    rng::AbstractRNG = Random.default_rng(),
    kwargs...
)

    # Extract parameter values from `fit`
    pars = if hasproperty(fit, :values)
        fit.values
    else
        fit
    end

    # Build Dirac priors at fitted values
    priors_fitted = Dict{Symbol, Distributions.Distribution}()
    for (k, v) in pairs(pars)
        ksym = k isa Symbol ? k : Symbol(k)
        priors_fitted[ksym] = Distributions.Dirac(v)
    end

    # Prepare data for simulation: replace outcome with missings
    keys_vec = collect(keys(data))
    data_for_sim = (; (
        k == outcome_name ? (k => fill(missing, length(getproperty(data, k)))) : (k => getproperty(data, k))
        for k in keys_vec
    )...)

    # Simulate outcomes with Dirac priors
    predicted = prior_sample(
        data_for_sim;
        model = model,
        n = n,
        priors = priors_fitted,
        outcome_name = outcome_name,
        rng = rng,
        kwargs...
    )

    # Compute log-likelihood of observed outcomes under fitted params
    # Using the original data (with observed outcome present)
    try
        m_obs = model(; data..., priors = priors_fitted, kwargs...)
        # Build NamedTuple of parameter values for loglikelihood
        pnames = collect(keys(priors_fitted))
        pvals = [mean(priors_fitted[p]) for p in pnames]
        params_nt = (; zip(pnames, pvals)...)
        ll = loglikelihood(m_obs, params_nt)
        return (; predicted, loglike = ll)
    catch
        return (; predicted, loglike = missing)
    end
end

"""
    posterior_predictive(
        df::AbstractDataFrame;
        model::Function,
        unpack_function::Function,
        fit,
        outcome_name::Symbol,
        n::Int64 = 1,
        rng::AbstractRNG = Random.default_rng(),
        kwargs...
    ) -> NamedTuple

Convenience wrapper for DataFrame inputs. Uses `unpack_function(df)` to map data
to the model, then calls `posterior_predictive(::NamedTuple, ...)`.
"""
function posterior_predictive(
    df::AbstractDataFrame;
    model::Function,
    unpack_function::Function,
    fit,
    outcome_name::Symbol,
    n::Int64 = 1,
    rng::AbstractRNG = Random.default_rng(),
    kwargs...
)
    data_nt = unpack_function(df)
    return posterior_predictive(
        data_nt;
        model = model,
        fit = fit,
        outcome_name = outcome_name,
        n = n,
        rng = rng,
        kwargs...
    )
end