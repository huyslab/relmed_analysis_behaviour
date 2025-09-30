using CairoMakie, AlgebraOfGraphics, DataFrames
using AlgebraOfGraphics: density

function plot_fixed_effects_recovery!(
    f::Union{Figure, GridPosition},
    chain::Chains,
    ground_truth::Dict
)

    # Exctract hyper-parameters from chain
    params = collect(keys(ground_truth))

    draws = DataFrame(chain[:, params, :])

    # Wide to long
    draws = stack(
        draws,
        Not([:iteration, :chain]),;
        variable_name = :parameter,
        value_name = :value
    )

    # Comput confidence intervals
    ci = combine(
        groupby(draws, :parameter),
        :value => (x -> quantile(x, 0.025)) => :lb,
        :value => (x -> quantile(x, 0.975)) => :ub
    )

    # Wide to long format for AoG
    ci = stack(
        ci,
        Not(:parameter); variable_name = :bound, value_name = :value
    )

    sort!(ci, [:parameter, :bound])

    ci.y .= 0.

    # Get true values from priors
    true_values = DataFrame(
        parameter = string.(params),
        value = [mean(ground_truth[k]) for k in params]
    )

    # Plot

    # Posterior density
    mp = data(draws) *
        mapping(
            :value,
            color = :chain => nonnumeric,
        ) * density(; datalimits=extrema) * visual(Lines)

    # True values
    mp += data(true_values) *
        mapping(
            :value,
        ) * visual(VLines, linestyle = :dash, color = :gray)

    # Confidence intervals
    mp += data(ci) *
        mapping(
            :value,
            :y,
        ) * visual(Lines, linewidth = 7, color = :gray)

    mp *= mapping(layout = :parameter)
    

    axs = draw!(f[1,1], mp, facet = (; linkxaxes = :none, linkyaxes = :none);
        axis = (; xlabel = ""))

    hideydecorations!.(axs)
    hidespines!.([ax.axis for ax in axs], :l)
    f

end

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

    # Summarize posterior
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

    ci.color = ifelse.(
        ci.lb .<= ci.value .<= ci.ub,
        :blue,
        :red
    )

    sort!(ci, :median)

    ci.y = 1:nrow(ci)

    # Plot
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

    # Add manual legend
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

    hideydecorations!.(ax)
    hidespines!(ax[1].axis, :l)

    rowgap!(gl, 1, 3)

    f

end