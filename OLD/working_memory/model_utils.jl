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

	for _ in 1:n_starts
		# Optimize
        try
            fit = Turing.Optimisation.estimate_mode(
                model,
                estimator
            )
            if fit.lp > best_lp
                best_lp = fit.lp
                best_result = fit
            end
        catch
            continue
        end
	end

	return best_result
end

function optimize(
	data::AbstractDataFrame;
    initial::Union{Float64, Nothing} = nothing,
    model::Function = RL_ss,
	estimate::String = "MAP",
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    parameters::Vector{Symbol} = collect(keys(priors)),
    gq_struct::Union{Nothing, AbstractDataFrame} = nothing, # Generate quantities based on set structure?
    fit_only::Bool = false,
    covs::Bool = false,
    n_starts::Int64 = 5
)
    data_for_fit = unpack_data(data)
    data_for_gq = isnothing(gq_struct) ? data_for_fit : unpack_data(gq_struct)
    
    data_model = model(
        data_for_fit,
        data.choice;
        priors = priors,
        initial = initial
    )

    if estimate == "MLE"
		fit = multistart_mode_estimator(data_model; estimator = MLE(), n_starts = n_starts)
	elseif estimate == "MAP"
		fit = multistart_mode_estimator(data_model; estimator = MAP(), n_starts = n_starts)
	end

    if fit_only || isnothing(fit)
        return fit
    end

    # Estimate covariance matrix?
    cov_mat = Matrix{Float64}(undef, length(parameters), length(parameters))
    if covs
        try
            cov_mat = vcov(fit)
        catch
            println("Covariance matrix could not be estimated.")
        end
    end

    # get predictions from the model for choices by setting a Dirac prior on the parameters
    priors_fitted = Dict{Symbol, Distribution}()
    for p in parameters
        priors_fitted = merge!(priors_fitted, Dict(p => Dirac(fit.values[p])))
    end
    
    gq_mod = model(
        data_for_gq,
        fill(missing, length(data_for_gq.block));
        priors = priors_fitted,
        initial = initial
    )

    # Draw parameters and simulate choice
    gq_samp = sample(
        Random.default_rng(),
        gq_mod,
        Prior(),
        1
    )
    gq = generated_quantities(gq_mod, gq_samp)
    choices = gq[1].choice
    loglike = gq[1].loglike
    bic = -2 * loglike + length(parameters) * log(nrow(data))

    return (est=fit, choices=choices, loglike=loglike, bic=bic, cov_mat=cov_mat)
end

# Find MLE / MAP multiple times
function optimize_multiple(
	data::AbstractDataFrame;
    initial::Union{Float64, Nothing} = nothing,
    model::Function = RL_ss,
    estimate::String = "MAP",
	include_true::Bool = true, # Whether to return true value if this is simulation
    priors::Dict = Dict(
        :ρ => truncated(Normal(0., 1.), lower = 0.),
        :a => Normal(0., 0.5)
    ),
    transformed::Dict{Symbol, Symbol} = Dict(:a => :α), # Transformed parameters
    parameters::Vector{Symbol} = collect(keys(priors)),
    gq_struct::Union{Nothing, AbstractDataFrame} = nothing, # Generate quantities based on set structure?
    n_starts::Int64 = 5,
    fit_only::Bool = false,
    covs::Bool = false
)
	ests = []
    choice_df = Dict{Int64, DataFrame}()
    cov_mat_dict = Dict{Int64, Matrix{Float64}}()
	lk = ReentrantLock()

	Threads.@threads for p in unique(data.PID)

		# Select data
		gdf = filter(x -> x.PID == p, data)

		# Optimize
		res = optimize(
			gdf;
			initial = initial,
            model = model,
			estimate = estimate,
            priors = priors,
            gq_struct = gq_struct,
            parameters = parameters,
            n_starts = n_starts,
            fit_only = fit_only,
            covs = covs
		)

        if isnothing(res)
            continue
        end

		# Return
        est = fit_only ? res : res.est
        est_dct = Dict{Symbol, Union{Int64, Float64}}(:PID => gdf.PID[1])
        est_par = Dict{Symbol, Symbol}()
        if include_true
            for p in parameters
                est_par[p] = haskey(transformed, p) ? transformed[p] : p
                est_dct[Symbol("true_$(est_par[p])")] = gdf[!, est_par[p]][1]
                est_dct[Symbol("MLE_$(est_par[p])")] = haskey(transformed, p) ? a2α(est.values[p]) : est.values[p]
            end
        else
            for p in parameters
                est_par[p] = haskey(transformed, p) ? transformed[p] : p
                est_dct[est_par[p]] = haskey(transformed, p) ? a2α(est.values[p]) : est.values[p]
            end
        end

        if !fit_only
            est_dct[:loglike], est_dct[:BIC] = res.loglike, res.bic
            gqf = isnothing(gq_struct) ? gdf : gq_struct
            cdf = DataFrame(
                block = gqf.block,
                valence = gqf.valence,
                trial = gqf.trial,
                pair = gqf.pair,
                predicted_choice = res.choices
            )
            if isnothing(gq_struct)
                insertcols!(cdf, :true_choice => gdf.choice)
            else
                ## account for early stopped blocks in PPCs
                pdf = DataFrame(
                    block = gdf.block,
                    valence = gdf.valence,
                    trial = gdf.trial,
                    pair = gdf.pair,
                    true_choice = gdf.choice
                )
                cdf = leftjoin(cdf, pdf, on = [:block, :valence, :trial, :pair])
            end
            insertcols!(cdf, 1, :PID => p)
        end

		lock(lk) do
            est = NamedTuple{Tuple(keys(est_dct))}(values(est_dct))
			push!(ests, est)
            if !fit_only
                if covs
                    merge!(cov_mat_dict, Dict(p => res.cov_mat))
                end
                choice_df[p] = cdf
            end
		end
	end

    if fit_only
        return DataFrame(ests)
    elseif !covs
        return DataFrame(ests), vcat(values(choice_df)...)
    else
        return DataFrame(ests), vcat(values(choice_df)...), cov_mat_dict
    end
end

function optimize_multiple_by_factor(
	df::DataFrame;
	model::Function,
	factor::Union{Symbol, Vector{Symbol}},
	priors::Dict,
    kwargs...
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
            priors = priors,
            fit_only = true,
            kwargs...
		)

		# Add factor variables
		factor_pairs = [col => gdf[!, col][1] for col in factor]

		insertcols!(fit, 1, factor_pairs...)

		push!(fits, fit)
	end

	# Combine to single DataFrame
	# fits = vcat(fits...)

	# # Sort
	# DataFrames.sort!(fits, vcat(factor, [:prolific_pid]))

	return fits

end