# Generate dashboard figures for PILT, WM, tests

# Setup 
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase, Tidier

function plot_learning_curves_by_factor!(
    f::Figure,
    df::DataFrame;
    factor::Symbol = :session,
    xcol::Symbol = :trial,
    early_stopping_at::Union{Int, Nothing} = 5,
    participant_id_column::Symbol = :participant_id
)

    # Remove non-response trials
    filter!(x -> x.response != "noresp", df)

    # Summarize by participant and trial
    acc_curve = combine(
        groupby(df, [participant_id_column, factor, xcol]),
        :response_optimal => mean => :acc
    )

    sort!(acc_curve, [participant_id_column, factor, xcol])

    # Summarize by trial
    acc_curve_sum = combine(
        groupby(acc_curve, [factor, xcol]),
        :acc => mean => :acc
    )

    # Plot
    mp = ((data(acc_curve) * mapping(
        xcol => "Trial #",
        :acc => "Prop. optimal choice",
        group = participant_id_column,
        color = participant_id_column,
    ) * visual(Lines, linewidth = 1, alpha = 0.7)) +
    (data(acc_curve_sum) * 
    mapping(
        xcol => "Trial #",
        :acc => "Prop. optimal choice",
    ) * visual(Lines, linewidth = 4))) * mapping(layout = factor)

    if early_stopping_at !== nothing
        mp = mp + mapping([early_stopping_at]) * visual(VLines, color = :grey, linestyle = :dash)
    end


    draw!(f, mp; axis = (; yticks = 0.:0.25:1.))

    return f
end

function compute_delays(vec::AbstractVector)
    last_seen = Dict{Any, Int}()
    delays = zeros(Int, length(vec))

    for (i, val) in enumerate(vec)
        delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
        last_seen[val] = i
    end

    return delays
end

# Prepare WM data for plotting
function prepare_WM_data(
    df::AbstractDataFrame;
    participant_id_column::Symbol = :participant_id,
)
    # Clean data
    data_clean = copy(df)

    # Sort
    sort!(
        data_clean,
        [participant_id_column, :session, :block, :trial]
    )

    # Apperance number
    transform!(
        groupby(data_clean, [participant_id_column, :session, :block, :stimulus_group]),
        :trial => (x -> 1:length(x)) => :appearance
    )

    # Compute delays
    DataFrames.transform!(
        groupby(
            data_clean,
            participant_id_column
        ),
        :stimulus_group => compute_delays => :delay,
    ) 

    # Remove non-response trials
    data_clean = filter(x -> x.response != "noresp", data_clean)

    # Previous correct
    DataFrames.transform!(
        groupby(
            data_clean,
            [participant_id_column, :stimulus_group]
        ),
        :response_optimal => lag => :previous_optimal,
    )

    return data_clean

end