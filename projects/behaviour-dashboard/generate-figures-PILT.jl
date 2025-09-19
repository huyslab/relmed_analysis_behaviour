# Generate dashboard figures for PILT

# Setup 
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase

function plot_learning_curves_by_factor!(
    f::Figure,
    df::DataFrame;
    factor::Symbol = :session,
    participant_id_column::Symbol = :participant_id
)

    # Remove non-response trials
    filter!(x -> x.response != "noresp", df)

    # Summarize by participant and trial
	acc_curve = combine(
		groupby(df, [participant_id_column, factor, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [participant_id_column, factor, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, [factor, :trial]),
		:acc => mean => :acc
	)

	# Plot
	mp = ((data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = participant_id_column,
		color = participant_id_column,
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
	) * visual(Lines, linewidth = 4))) * mapping(layout = factor) +
    mapping([5]) * visual(VLines, color = :grey, linestyle = :dash)


	draw!(f, mp; axis = (; yticks = 0.:0.25:1.))

    return f
end

