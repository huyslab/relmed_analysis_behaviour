using DataFrames, CairoMakie, AlgebraOfGraphics, StatsBase

function plot_pavlovian_lottery_rt!(
    f::Figure,
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    facet::Symbol = :session,
    config::Dict = Dict(
        :individual_alpha => 0.5,
        :sms => 10,
        :ms => 20,
        :lw => 2
    )
)

    # Remove missing responses
    df_clean = filter(row -> !ismissing(row.rt) && !isnothing(row.rt), df)

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

    sort!(rt_sum_sum, [:pavlovian_value, facet]),
    sort!(rt_sum, [:pavlovian_value, facet])

    # Plot    
    mp = data(rt_sum) *
    mapping(
        :pavlovian_value => nonnumeric,
        :rt,
    ) * visual(Scatter, markersize = config[:sms], alpha = config[:individual_alpha])

    mp *= mapping(color = experiment.participant_id_column)

    mp += data(rt_sum_sum) *
    (mapping(
        :pavlovian_value => nonnumeric,
        :rt,
        :se
    ) * visual(Errorbars, linewidth = lw, color = :black) +
    mapping(
        :pavlovian_value => nonnumeric,
        :rt,
    ) * visual(ScatterLines, linewidth = config[:lw], markersize = config[:ms], color = :black))


    mp *= mapping(layout = facet)

    draw!(f[1, 1], mp, axis = (; xlabel = "Pavlovian Value", ylabel = "Reaction Time (ms)"))
    f

end