"""
Card Choosing Task Figure Generation

This module provides functions for generating dashboard figures for behavioral experiments,
specifically for Probabilistic Instrumental Learning Task (PILT) and Working Memory (WM) tasks.

Key functions:
- plot_learning_curves_by_facet!: Plots learning curves with individual and average trajectories
- plot_learning_curves_by_color_facet!: Plots learning curves grouped by color/condition
- prepare_WM_data: Preprocesses working memory data for analysis
- plot_learning_curve_by_delay_bins!: Plots learning curves binned by delay intervals
- compute_delays: Utility function to calculate delays between stimulus presentations

The plots show proportion of optimal choices over trials/appearances, with options for
displaying individual participant data or standard error bands.
"""

# Setup 
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase

"""
    plot_learning_curves_by_facet!(f::Figure, df::DataFrame; kwargs...)

Plot learning curves showing proportion of optimal choices over trials, 
with individual participant trajectories and group averages.

# Arguments
- `f::Figure`: Makie figure to draw into
- `df::DataFrame`: Data containing response and trial information
- `facet::Symbol`: Column to facet by (default: :session)
- `xcol::Symbol`: X-axis column (default: :trial)
- `early_stopping_at::Union{Int, Nothing}`: Add vertical line at trial number (default: 5)
- `participant_id_column::Symbol`: Column identifying participants (default: :participant_id)
"""
function plot_learning_curves_by_facet!(
    f::Figure,
    df::DataFrame;
    facet::Symbol = :session,
    xcol::Symbol = :trial,
    early_stopping_at::Union{Int, Nothing} = 5,
    participant_id_column::Symbol = :participant_id,
    config::Dict = plot_config
)

    # Remove non-response trials to focus on actual choices
    filter!(x -> x.response != "noresp", df)

    # Calculate accuracy (proportion optimal) for each participant per trial
    acc_curve = combine(
        groupby(df, [participant_id_column, facet, xcol]),
        :response_optimal => mean => :acc
    )

    sort!(acc_curve, [participant_id_column, facet, xcol])

    # Calculate group averages across participants
    acc_curve_sum = combine(
        groupby(acc_curve, [facet, xcol]),
        :acc => mean => :acc
    )

    # Create the plot mapping with individual trajectories and group average
    mp = ((data(acc_curve) * mapping(
        xcol => "Trial #",
        :acc => "Prop. optimal choice",
        group = participant_id_column,
        color = participant_id_column,
    ) * visual(Lines, linewidth = config[:individual_linewidth], alpha = config[:individual_alpha])) +
    (data(acc_curve_sum) * 
    mapping(
        xcol => "Trial #",
        :acc => "Prop. optimal choice",
    ) * visual(Lines, linewidth = config[:group_linewidth]))) * mapping(layout = facet)

    # Add vertical line for early stopping indicator if specified
    if early_stopping_at !== nothing
        mp = mp + mapping([early_stopping_at]) * visual(VLines, color = :grey, linestyle = :dash)
    end

    # Draw the plot with custom y-axis ticks
    draw!(f, mp; axis = (; yticks = 0.:0.25:1.))

    return f
end

"""
    plot_learning_curves_by_color_facet!(f::Figure, df::DataFrame; kwargs...)

Plot learning curves with color-coded conditions, showing either individual trajectories
or standard error bands based on the variability parameter.

# Arguments
- `f::Figure`: Makie figure to draw into
- `df::DataFrame`: Data containing response and trial information
- `facet::Symbol`: Column to facet by (default: :session)
- `xcol::Symbol`: X-axis column (default: :trial)
- `color::Symbol`: Column to color by (default: :valence)
- `color_label::String`: Legend label for color mapping (default: "Valence")
- `early_stopping_at::Union{Int, Nothing}`: Add vertical line at trial number (default: 5)
- `participant_id_column::Symbol`: Column identifying participants (default: :participant_id)
- `variability::Symbol`: Display variability as :se (standard error) or :individuals (default: :se)
"""
function plot_learning_curves_by_color_facet!(
    f::Figure,
    df::DataFrame;
    facet::Symbol = :session,
    xcol::Symbol = :trial,
    color::Symbol = :valence,
    color_label::String = "Valence",
    early_stopping_at::Union{Int, Nothing} = 5,
    participant_id_column::Symbol = :participant_id,
    variability::Symbol = :se, # :se or :individuals
    config::Dict = plot_config
)

    # Remove non-response trials
    filter!(x -> x.response != "noresp", df)

    # Summarize by participant and trial
    acc_curve = combine(
        groupby(df, [participant_id_column, facet, color, xcol]),
        :response_optimal => mean => :acc
    )

    sort!(acc_curve, [participant_id_column, facet, color, xcol])

    # Create unique group identifiers for individual trajectories
    acc_curve.group = string.(acc_curve.participant_id) .* "_" .* string.(acc_curve[!, color])

    # Summarize by trial with standard error bounds
    acc_curve_sum = combine(
        groupby(acc_curve, [facet, color, xcol]),
        :acc => mean => :acc,
        :acc => (x -> mean(x) - sem(x)) => :lb,
        :acc => (x -> mean(x) + sem(x)) => :ub
    )

    # Create plot mapping based on variability display option
    if variability == :individuals
        mp = (data(acc_curve) * mapping(
            xcol,
            :acc,
            group = :group,
            linestyle = participant_id_column,
            color = color => color_label
        ) * visual(Lines, linewidth = config[:individual_linewidth], alpha = config[:individual_alpha]))
    elseif variability == :se
        mp = data(acc_curve_sum) *
        mapping(
            xcol,
            :lb,
            :ub,
            color = color => color_label
        ) * visual(Band, alpha = config[:band_alpha])
    end


    mp += data(acc_curve_sum) * 
    mapping(
        xcol,
        :acc,
        color = color => color_label
    ) * visual(Lines, linewidth = config[:group_linewidth])

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

"""
    compute_delays(vec::AbstractVector)

Compute delays between stimulus presentations for working memory analysis.
Returns the number of trials since each stimulus was last seen.

# Arguments
- `vec::AbstractVector`: Vector of stimulus identifiers

# Returns
- `Vector{Int}`: Delay (in trials) since each stimulus was last presented, 0 for first occurrence
"""
function compute_delays(vec::AbstractVector)
    last_seen = Dict{Any, Int}()  # Track last position of each stimulus
    delays = zeros(Int, length(vec))

    for (i, val) in enumerate(vec)
        # Calculate delay: current position - last seen position (or 0 if first occurrence)
        delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
        last_seen[val] = i  # Update last seen position
    end

    return delays
end

# Prepare WM data for plotting
"""
    prepare_WM_data(df::AbstractDataFrame; participant_id_column::Symbol = :participant_id)

Preprocess working memory data for analysis by adding appearance numbers, 
computing delays between stimulus presentations, and cleaning the dataset.

# Arguments
- `df::AbstractDataFrame`: Raw behavioral data
- `participant_id_column::Symbol`: Column identifying participants (default: :participant_id)

# Returns
- Cleaned DataFrame with added columns: :appearance, :delay, :previous_optimal
"""
function prepare_WM_data(
    df::AbstractDataFrame;
    participant_id_column::Symbol = :participant_id,
)
    # Clean data
    data_clean = copy(df)

    # Sort by participant, session, block, and trial for proper ordering
    sort!(
        data_clean,
        [participant_id_column, :session, :block, :trial]
    )

    # Add appearance number (how many times each stimulus has been seen)
    transform!(
        groupby(data_clean, [participant_id_column, :session, :block, :stimulus_group]),
        :trial => (x -> 1:length(x)) => :appearance
    )

    # Compute delays between stimulus presentations
    DataFrames.transform!(
        groupby(
            data_clean,
            participant_id_column
        ),
        :stimulus_group => compute_delays => :delay,
    ) 

    # Remove non-response trials for analysis
    data_clean = filter(x -> x.response != "noresp", data_clean)

    # Add previous trial outcome for each stimulus
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
"""
    plot_learning_curve_by_delay_bins!(f::Figure, df::DataFrame; kwargs...)

Plot learning curves for working memory analysis, showing proportion of optimal choices
by stimulus appearance number, grouped by delay bins (time since stimulus last seen).

# Arguments
- `f::Figure`: Makie figure to draw into
- `df::DataFrame`: Preprocessed working memory data (use prepare_WM_data first)
- `participant_id_column::Symbol`: Column identifying participants (default: :participant_id)
- `facet::Symbol`: Column to facet by (default: :session)
- `variability::Symbol`: Display variability as :se (standard error) or :individuals (default: :se)
- `lw::Real`: Line width for main lines (default: 4)
- `tlw::Real`: Line width for individual trajectories (default: 1)
- `ms::Real`: Marker size for main points (default: 20)
- `sms::Real`: Marker size for individual points (default: 4)
"""
function plot_learning_curve_by_delay_bins!(
    f::Figure,
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    facet::Symbol = :session,
    variability::Symbol = :se, # :se or :individuals
    config::Dict = plot_config
)
	
    # Recode delay into meaningful bins for analysis
    recoder = (x, edges, labels) -> ([let idx = findfirst(v ≤ edge for edge in edges); idx === nothing ? labels[end] : labels[idx] end for v in x])
	
    df.delay_bin = recoder(df.delay, [0, 1, 5, 10], ["0", "1", "2-5", "6-10"])
	
	# Summarize by participant - calculate accuracy for each appearance
	app_curve = combine(
		groupby(df, [participant_id_column, facet, :delay_bin, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize across participants - get group means and standard errors
	app_curve_sum = combine(
		groupby(app_curve, [facet, :delay_bin, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute confidence bounds for error bands
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort for proper plotting order
	sort!(app_curve_sum, [facet, :delay_bin, :appearance])
    sort!(app_curve, [facet, participant_id_column, :delay_bin, :appearance])

	# Create mapping based on display preference
    if variability == :se
        # Show error bands and error bars for delay=0 (first presentations)
        mp = (data(app_curve_sum) * (
            mapping(
                :appearance,
                :lb,
                :ub,
                color = :delay_bin  => "Delay",
                layout = facet
        ) * visual(Band, alpha = config[:band_alpha]) +
            mapping(
                :appearance,
                :acc => "Prop. optimal choice",
                color = :delay_bin  => "Delay",
                col = facet
        ) * visual(Lines; linewidth = config[:group_linewidth]))) + (
            data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
            (mapping(
                :appearance ,
                :acc,
                :se,
                color = :delay_bin => "Delay",
                col = facet
            ) * visual(Errorbars, linewidth = config[:errorbar_linewidth]) +
            mapping(
                :appearance ,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = config[:large_markersize]))
        )
    elseif variability == :individuals
        # Show individual participant trajectories
        mp = data(filter(x -> x.delay_bin != "0", app_curve)) *
        mapping(
                :appearance,
                :acc,
                color = :delay_bin  => "Delay",
                group = participant_id_column,
                col = facet
        ) * visual(Lines; linewidth = config[:thin_linewidth], linestyle = :dash) +
        data(filter(x -> x.delay_bin == "0", app_curve)) *
        mapping(
                :appearance,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = config[:small_markersize]) +
        data(filter(x -> x.delay_bin != "0", app_curve_sum)) *
        mapping(
                :appearance,
                :acc => "Prop. optimal choice",
                color = :delay_bin  => "Delay",
                col = facet
        ) * visual(Lines; linewidth = config[:group_linewidth]) +
        data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
        mapping(
                :appearance ,
                :acc,
                color = :delay_bin  => "Delay",
                col = facet
            ) * visual(Scatter, markersize = config[:large_markersize])
    end

	# Draw the plot with axis labels
	plt = draw!(f[1,1], mp; 
		axis=(; 
            xlabel = "Appearance #",
			ylabel = "Prop. optimal choice ±SE"
		)
	)

	# Add horizontal legend at top
	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end
