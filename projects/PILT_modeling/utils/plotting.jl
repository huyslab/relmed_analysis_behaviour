"""
Plotting utilities for model diagnostics and parameter recovery analysis.
"""

using CairoMakie, AlgebraOfGraphics, DataFrames
using AlgebraOfGraphics: density

"""
    plot_fixed_effects_recovery!(f, chain, ground_truth)

Plot posterior densities, confidence intervals, and true values for fixed effects.

# Arguments
- `f`: Figure or GridPosition to plot into
- `chain::Chains`: MCMC samples
- `ground_truth::Dict`: True parameter values

# Returns
Modified figure with density plots and recovery diagnostics
"""
function plot_fixed_effects_recovery!(
    f::Union{Figure, GridPosition},
    chain::Chains,
    ground_truth::Dict
)
    # Extract hyperparameters from chain
    params = collect(keys(ground_truth))

    # Extract true parameter values, handling array distributions
    parameter_names = String[]
    parameter_values = Float64[]
    
    for param in params
        mean_val = mean(ground_truth[param])
        if mean_val isa AbstractVector
            # Handle array distributions - append [i] to parameter names
            for (i, val) in enumerate(mean_val)
                push!(parameter_names, "$(param)[$(i)]")
                push!(parameter_values, val)
            end
        else
            # Handle scalar distributions
            push!(parameter_names, string(param))
            push!(parameter_values, mean_val)
        end
    end
    
    true_values = DataFrame(
        parameter = parameter_names,
        value = parameter_values
    )

    # Check which parameters exist in the chain
    chain_params = names(chain)
    missing_params = setdiff(Symbol.(parameter_names), chain_params)
    available_params = intersect(Symbol.(parameter_names), chain_params)
    
    if !isempty(missing_params)
        @warn "The following parameters are missing from the chain and will be excluded from plotting: $(missing_params)"
    end
    
    # Filter true_values to only include available parameters
    true_values = filter(row -> row.parameter in string.(available_params), true_values)

    draws = DataFrame(chain[:, available_params, :])

    # Convert to long format
    draws = stack(
        draws,
        Not([:iteration, :chain]),;
        variable_name = :parameter,
        value_name = :value
    )

    # Compute 95% confidence intervals
    ci = combine(
        groupby(draws, :parameter),
        :value => (x -> quantile(x, 0.025)) => :lb,
        :value => (x -> quantile(x, 0.975)) => :ub
    )

    # Reshape for plotting
    ci = stack(
        ci,
        Not(:parameter); variable_name = :bound, value_name = :value
    )
    sort!(ci, [:parameter, :bound])
    ci.y .= 0.

    # Create plot layers
    # Posterior density curves
    mp = data(draws) *
        mapping(
            :value,
            color = :chain => nonnumeric,
        ) * density(; datalimits=extrema) * visual(Lines)

    # True values as vertical dashed lines
    mp += data(true_values) *
        mapping(
            :value,
        ) * visual(VLines, linestyle = :dash, color = :gray)

    # Confidence intervals as horizontal lines
    mp += data(ci) *
        mapping(
            :value,
            :y,
        ) * visual(Lines, linewidth = 7, color = :gray)

    mp *= mapping(layout = :parameter)
    

    axs = draw!(f[1,1], mp, facet = (; linkxaxes = :none, linkyaxes = :none);
        axis = (; xlabel = ""))

    # Clean up axes
    hideydecorations!.(axs)
    hidespines!.([ax.axis for ax in axs], :l)
    f

end

"""
    plot_random_effects_recovery!(f, chain, true_values)

Plot recovery diagnostics for random effects with confidence intervals.

# Arguments
- `f`: Figure or GridPosition to plot into  
- `chain::Chains`: MCMC samples
- `true_values::AbstractDataFrame`: True random effect values

# Returns
Modified figure with caterpillar plot showing recovery performance
"""

function plot_random_effects_recovery!(
    f::Union{Figure, GridPosition},
    chain::Chains,
    true_values::AbstractDataFrame
)
    # Extract fitted values from chain
    draws = DataFrame(chain[:, Symbol.(true_values.parameter), :])
    draws = stack(
        draws,
        Not([:iteration, :chain]); variable_name = :parameter
    )

    # Summarize posterior with quantiles
    ci = combine(
        groupby(
            draws,
            :parameter
        ),
        :value => median => :median,
        :value => (x -> median(x) - quantile(x, 0.025)) => :lle,
        :value => (x -> quantile(x, 0.975) - median(x)) => :uue,
        :value => (x -> quantile(x, 0.025)) => :lb,
        :value => (x -> quantile(x, 0.975)) => :ub,
        :value => (x -> median(x) - quantile(x, 0.25)) => :le,
        :value => (x -> quantile(x, 0.75) - median(x)) => :ue
    )

    leftjoin!(ci, true_values, on = :parameter)

    # Color code based on coverage
    ci.color = ifelse.(
        ci.lb .<= ci.value .<= ci.ub,
        :blue,
        :red
    )

    sort!(ci, :median)
    ci.y = 1:nrow(ci)

    # Create caterpillar plot layers
    mp = mapping(
        :median,
        :y,
        :lle,
        :uue
    ) * visual(Errorbars, color = :grey, linewidth = 0.5, direction = :x)

    mp += mapping(
        :median,
        :y,
        :le,
        :ue
    ) * visual(Errorbars, color = :grey, linewidth = 1.5, direction = :x)

    mp += mapping(
        :median,
        :y
    ) * visual(Scatter, color = :gray)

    mp += mapping(
        :value,
        :y,
        color = :color => verbatim
    ) * visual(Scatter, marker = :x)

    mp *= data(ci)

    gl = f[1,1] = GridLayout()
    ax = draw!(gl[3,1], mp; axis = (; xlabel = "Random effect value"))

    # Add legends
    Legend(gl[1, 1], 
        [MarkerElement(color = :gray, marker = :circle),
         LineElement(color = :gray, linewidth = 0.5*2),
         LineElement(color = :gray, linewidth = 1.5*2)
        ],
        ["Posterior median", "95% PI", "50% PI"],
        orientation = :horizontal,
        tellwidth = false,
        framevisible = false,
        padding = 0.
    )

    Legend(gl[2, 1], 
        [MarkerElement(color = :blue, marker = :x),
         MarkerElement(color = :red, marker = :x)],
        ["within 95% PI", "outside 95% PI"],
        "True value",
        orientation = :horizontal,
        tellwidth = false,
        framevisible = false,
        titleposition = :left,
        titlefont = :regular,
        padding = 0.
    )

    # Clean up axes
    hideydecorations!.(ax)
    hidespines!(ax[1].axis, :l)
    rowgap!(gl, 1, 3)
    f
end