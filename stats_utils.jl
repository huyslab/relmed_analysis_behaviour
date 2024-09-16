### General stats utility functions

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

# Bootstrap a correlation
function bootstrap_correlation(x, y, n_bootstrap=1000)
	n = length(x)
	corrs = Float64[]  # To store the bootstrap correlations

	for i in 1:n_bootstrap
		# Resample the data with replacement
		idxs = sample(Xoshiro(i), 1:n, n, replace=true)
		x_resample = x[idxs]
		y_resample = y[idxs]
		
		# Compute the correlation for the resampled data
		push!(corrs, cor(x_resample, y_resample))
	end

	return corrs
end