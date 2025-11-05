using DataFrames, CairoMakie, AlgebraOfGraphics, StatsBase

function plot_pavlovian_lottery_rt!(
    f::Figure,
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    facet::Symbol = :session,
    config::Dict = plot_config
)

    # Remove missing responses
    df_clean = filter(row -> !ismissing(row.rt) && !isnothing(row.rt), df)

    # Remove outlier RTs due to missing responses
    filter!(row -> row.rt <= 4000, df_clean)

    # Summarize RT by participant, pavlovian value, and facet
    rt_sum = combine(
        groupby(df_clean, [experiment.participant_id_column, :pavlovian_value, facet]),
        :rt => mean => :rt,
        :rt => sem => :se
    )

    # Summarize by pavlovian value and facet
    rt_sum_sum = combine(
        groupby(rt_sum, [:pavlovian_value, facet]),
        :rt => mean => :rt,
        :rt => sem => :se
    )

    sort!(rt_sum_sum, [:pavlovian_value, facet])
    sort!(rt_sum, [:pavlovian_value, facet])

    # Plot    
    mp = data(rt_sum) *
    mapping(
        :pavlovian_value => nonnumeric,
        :rt,
    ) * visual(Scatter, markersize = config[:small_markersize], alpha = config[:scatter_alpha])

    mp *= mapping(color = experiment.participant_id_column)

    mp += data(rt_sum_sum) *
    (mapping(
        :pavlovian_value => nonnumeric,
        :rt,
        :se
    ) * visual(Errorbars, linewidth = config[:thick_linewidth], color = :black) +
    mapping(
        :pavlovian_value => nonnumeric,
        :rt,
    ) * visual(ScatterLines, linewidth = config[:thick_linewidth], markersize = config[:large_markersize], color = :black))


    mp *= mapping(layout = facet)

    draw!(f[1, 1], mp, axis = (; xlabel = "Pavlovian Value", ylabel = "Reaction Time (ms)"))
    f

end