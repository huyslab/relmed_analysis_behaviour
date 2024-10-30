# This file contains general functions to work with models

## MLE / MAP estimation -------------------------------------------------------------------------
"""
    multistart_mode_estimator(model::Turing.Model; estimator::Union{MLE, MAP}, n_starts::Int64 = 5) -> OptimizationResult

Estimates the mode of a Turing model using a multistart approach. 

This function runs the specified estimator (`MLE` or `MAP`) multiple times (`n_starts`) to find the optimal mode, aiming to avoid local optima. It returns the result with the highest log-probability among the runs.

# Arguments
- `model`: A `Turing.Model` to be optimized.
- `estimator`: The mode estimation method, either `MLE` or `MAP`.
- `n_starts`: The number of optimization starts, defaulting to 5.

# Returns
- `best_result`: The optimization result with the highest log-probability.
"""
function multistart_mode_estimator(
	model::Turing.Model;
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
    optimize(data::NamedTuple; initial::Union{Float64, Nothing} = nothing, model::Function, estimate::String = "MAP", initial_params::Union{AbstractVector, Nothing} = nothing, priors::Dict) -> OptimizationResult

Fits a probabilistic model to the provided data using the specified mode estimation method.

This function takes a data set (formatted as NamedTuple) and a model and estimates the mode using either Maximum Likelihood Estimation (MLE) or Maximum A Posteriori (MAP) estimation. It returns the optimization result with the highest log-probability.

# Arguments
- `data`: A `NamedTuple` containing the data for the model. `dv` Should be present with the dependent variable for the model.
- `model`: A function representing the model to be optimized.
- `estimate`: A string specifying the estimation method (`"MLE"` or `"MAP"`), defaulting to `"MAP"`.
- `priors`: A dictionary of prior distributions for model parameters.

# Returns
- `fit`: The result of the mode estimation with the highest log-probability.
"""
function optimize(
	data::NamedTuple;
	model::Function,
	estimate::String = "MAP",
    priors::Dict,
	n_starts::Int64 = 5
)
	
	# Pass data to model
    data_model = model(;
        data...,
        priors = priors
    )

	if estimate == "MLE"
		fit = multistart_mode_estimator(data_model; estimator = MLE(), n_starts = n_starts)
	elseif estimate == "MAP"
		fit = multistart_mode_estimator(data_model; estimator = MAP(), n_starts = n_starts)
	end

	return fit
end

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


## Fisher Information functions -----------------------------------------------------------------

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
    FI(data::DataFrame, model::Function, map_data_to_model::Function, param_names::Vector{Symbol}, id_col::Symbol = :PID, kwargs...)

Compute the Fisher Information for multiple simulated datasets at the true parameter values used to generate the data.

### Arguments
- `data::DataFrame`: A DataFrame containing the simulated datasets. Each group (split by `id_col`) is treated as an individual dataset.
- `model::Function`: A Turing model.
- `map_data_to_model::Function`: A function that maps an AbstractDataFrame to a NamedTuple of arguments to be passed to `model`.
- `param_names::Vector{Symbol}`: A vector of symbols representing the names of the parameters for which Fisher Information is computed.
- `id_col::Symbol`: The column name used to split the dataset into different groups. Default is `:PID`.
- `kwargs...`: Additional keyword arguments passed to the `model` function.

### Returns
- Returns the sum of Fisher Information computed for each group in the dataset.
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
	for gdf in groupby(data, id_col)
		
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