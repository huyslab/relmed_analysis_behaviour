function preprocess_reversal_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )

	# Sort
	out_df = sort(out_df, [participant_id_column, :session, :block, :trial])

	# Cumulative trial number
	DataFrames.transform!(
		groupby(out_df, [participant_id_column, :session]),
		:trial => (x -> 1:length(x)) => :ctrial
	)

	# Exclude no response trials
	filter!(x -> !isnothing(x.response_optimal), out_df)

	# Auxillary variables --------------------------		
	# Number trials leading to reversal
	DataFrames.transform!(
		groupby(out_df, [participant_id_column, :session, :block]),
		:trial => (x -> x .- maximum(x) .- 1) => :trial_pre_reversal
	)

	# Count number of block
	DataFrames.transform!(
		groupby(out_df, [participant_id_column, :session]),
		:block => (x -> maximum(x)) => :n_blocks
	)

	# Create feedback_optimal and feedback_suboptimal
	out_df.feedback_optimal = ifelse.(
		out_df.optimal_right .== 1,
		out_df.feedback_right,
		out_df.feedback_left
	)

	out_df.feedback_suboptimal = ifelse.(
		out_df.optimal_right .== 0,
		out_df.feedback_right,
		out_df.feedback_left
	)

	@assert all(combine(
		groupby(out_df, [participant_id_column, :session]),
		:ctrial => issorted => :trial_sorted
	).trial_sorted)

	return out_df
end

function plot_reversal_accuracy_curve!(
    f::Figure,
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
    
	# Summarize accuracy pre reversal
	sum_pre = combine(
		groupby(
			filter(x -> (x.trial_pre_reversal > -4) && 
				(x.block < x.n_blocks) && (x.trial < 48), df), 
			[participant_id_column, :trial_pre_reversal]
		),
		:response_optimal => mean => :acc
	)

	rename!(sum_pre, :trial_pre_reversal => :trial)

	sum_sum_pre = combine(
		groupby(sum_pre, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> x.trial < 6, df),
			[participant_id_column, :trial]
		),
		:response_optimal => mean => :acc
	)

	sum_sum_post = combine(
		groupby(sum_post, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Concatenate pre and post
	sum_sum_pre_post = vcat(sum_sum_pre, sum_sum_post)
	sum_pre_post = vcat(sum_pre, sum_post)

	# Create group variable to break line plot
	sum_sum_pre_post.group = sign.(sum_sum_pre_post.trial)
	sum_pre_post.group = sign.(sum_pre_post.trial) .* 
		map(val -> findfirst(==(val), unique(sum_pre_post.prolific_pid)), 
			sum_pre_post.prolific_pid)

	# Color by accuracy on trial - 3
	DataFrames.transform!(
		groupby(sum_pre_post, participant_id_column),
		[:trial, :acc] => ((t, a) -> mean(a[t .== -3])) => :color
	)

	# Sort for plotting
	sort!(sum_pre_post, [participant_id_column, :trial])

	# Plot
	mp = data(sum_pre_post) *
		mapping(
			:trial => "Trial relative to reversal",
			:acc => "Prop. optimal choice",
			group = :group => nonnumeric,
			color = :color
		) * visual(Lines, linewidth = 1) +
		
	data(sum_sum_pre_post) *
		(
			mapping(
				:trial => "Trial relative to reversal",
				:acc  => "Prop. optimal choice",
				:se
			) * visual(Errorbars) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice"
			) * 
			visual(Scatter) +
			mapping(
				:trial => "Trial relative to reversal",
				:acc => "Prop. optimal choice",
				group = :group => nonnumeric 
			) * 
			visual(Lines)
		) +
		mapping([0]) * visual(VLines, color = :grey, linestyle = :dash)

	draw!(f[1,1], mp, scales(Color = (; colormap = :roma)); 
		axis = (; xticks = -3:5, yticks = 0:0.25:1.))

	f

end