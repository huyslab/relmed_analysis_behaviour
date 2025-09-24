# Generate dashboard figures for PILT, WM, tests

# Setup 
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase

function plot_learning_curves_by_facet!(
    f::Figure,
    df::DataFrame;
    facet::Symbol = :session,
    xcol::Symbol = :trial,
    early_stopping_at::Union{Int, Nothing} = 5,
    participant_id_column::Symbol = :participant_id
)

    # Remove non-response trials
    filter!(x -> x.response != "noresp", df)

    # Summarize by participant and trial
    acc_curve = combine(
        groupby(df, [participant_id_column, facet, xcol]),
        :response_optimal => mean => :acc
    )

    sort!(acc_curve, [participant_id_column, facet, xcol])

    # Summarize by trial
    acc_curve_sum = combine(
        groupby(acc_curve, [facet, xcol]),
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
    ) * visual(Lines, linewidth = 4))) * mapping(layout = facet)

    if early_stopping_at !== nothing
        mp = mp + mapping([early_stopping_at]) * visual(VLines, color = :grey, linestyle = :dash)
    end


    draw!(f, mp; axis = (; yticks = 0.:0.25:1.))

    return f
end

function plot_learning_curves_by_color_facet!(
    f::Figure,
    df::DataFrame;
    facet::Symbol = :session,
    xcol::Symbol = :trial,
    color::Symbol = :valence,
    color_label::String = "Valence",
    early_stopping_at::Union{Int, Nothing} = 5,
    participant_id_column::Symbol = :participant_id,
    variability::Symbol = :se # :se or :individuals
)

    # Remove non-response trials
    filter!(x -> x.response != "noresp", df)

    # Summarize by participant and trial
    acc_curve = combine(
        groupby(df, [participant_id_column, facet, color, xcol]),
        :response_optimal => mean => :acc
    )

    sort!(acc_curve, [participant_id_column, facet, color, xcol])

    acc_curve.group = acc_curve.participant_id .* "_" .* string.(acc_curve[!, color])

    # Summarize by trial
    acc_curve_sum = combine(
        groupby(acc_curve, [facet, color, xcol]),
        :acc => mean => :acc,
        :acc => (x -> mean(x) - sem(x)) => :lb,
        :acc => (x -> mean(x) + sem(x)) => :ub
    )

    # Plot
    if variability == :individuals
        mp = (data(acc_curve) * mapping(
            xcolfacet,
            :acc,
            group = :group,
            linestyle = participant_id_column,
            color = color => color_label
        ) * visual(Lines, linewidth = 1, alpha = 0.7))
    elseif variability == :se
        mp = data(acc_curve_sum) *
        mapping(
            xcol,
            :lb,
            :ub,
            color = color => color_label
        ) * visual(Band, alpha = 0.5)
    end


    mp += data(acc_curve_sum) * 
    mapping(
        xcol,
        :acc,
        color = color => color_label
    ) * visual(Lines, linewidth = 4)

    mp *= mapping(layout = facet)

    if early_stopping_at !== nothing
        mp = mp + mapping([early_stopping_at]) * visual(VLines, color = :grey, linestyle = :dash)
    end

    if variability == :individuals
        plt = draw!(f[1,1], mp, scales(LineStyle = (; legend = false)); 
            axis = (; 
                yticks = 0.:0.25:1.,
                xlabel = "Trial #",
                ylabel = "Prop. optimal choice"
            )
        )
    else
        plt = draw!(f[1,1], mp; 
            axis = (; 
                yticks = 0.:0.25:1.,
                xlabel = "Trial #",
                ylabel = "Prop. optimal choice ±SE"
            )
        )
    end

    legend!(
        f[0,1], 
        plt, 
        tellwidth = false, 
        halign = 0.5, 
        orientation = :horizontal, 
        framevisible = false)

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

    # Appearance number
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
        :response_optimal => (x -> length(x) == 0 ? eltype(x)[] : vcat([missing], x[1:end-1])) => :previous_optimal,
    )

    return data_clean

end

# Plot learning curve with delay bins
function plot_learning_curve_by_delay_bins!(
    f::Figure,
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    facet::Symbol = :session,
    variability::Symbol = :se, # :se or :individuals
    lw::Real = 4,
    tlw::Real = 1,
    ms::Real = 20,
    sms::Real = 4
)
	
    # Recode delay into bins
    recoder = (x, edges, labels) -> ([let idx = findfirst(v ≤ edge for edge in edges); idx === nothing ? labels[end] : labels[idx] end for v in x])
	
    df.delay_bin = recoder(df.delay, [0, 1, 5, 10], ["0", "1", "2-5", "6-10"])
	
	# Summarize by participant
	app_curve = combine(
		groupby(df, [participant_id_column, facet, :delay_bin, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	app_curve_sum = combine(
		groupby(app_curve, [facet, :delay_bin, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [facet, :delay_bin, :appearance])
    sort!(app_curve, [facet, participant_id_column, :delay_bin, :appearance])

	# Create mapping
    if variability == :se
        mp = (data(app_curve_sum) * (
            mapping(
                :appearance,
                :lb,
                :ub,
                color = :delay_bin  => "Delay",
                layout = facet
        ) * visual(Band, alpha = 0.5) +
            mapping(
                :appearance,
                :acc => "Prop. optimal choice",
                color = :delay_bin  => "Delay",
                col = facet
        ) * visual(Lines; linewidth = lw))) + (
            data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
            (mapping(
                :appearance ,
                :acc,
                :se,
                color = :delay_bin => "Delay",
                col = facet
            ) * visual(Errorbars, linewidth = lw) +
            mapping(
                :appearance ,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = ms))
        )
    elseif variability == :individuals
        mp = data(filter(x -> x.delay_bin != "0", app_curve)) *
        mapping(
                :appearance,
                :acc,
                color = :delay_bin  => "Delay",
                group = participant_id_column,
                col = facet
        ) * visual(Lines; linewidth = tlw, linestyle = :dash) +
        data(filter(x -> x.delay_bin == "0", app_curve)) *
        mapping(
                :appearance,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = sms) +
        data(filter(x -> x.delay_bin != "0", app_curve_sum)) *
        mapping(
                :appearance,
                :acc => "Prop. optimal choice",
                color = :delay_bin  => "Delay",
                col = facet
        ) * visual(Lines; linewidth = lw) +
        data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
        mapping(
                :appearance ,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = ms)
    end

	plt = draw!(f[1,1], mp; 
		axis=(; 
            xlabel = "Appearance #",
			ylabel = "Prop. optimal choice ±SE"
		)
	)

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end
