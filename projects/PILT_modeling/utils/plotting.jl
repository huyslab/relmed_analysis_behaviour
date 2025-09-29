using CairoMakie, AlgebraOfGraphics, DataFrames

function plot_hyper_parameters_vs_fitted!(
    f::Union{Figure, GridLayout},
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