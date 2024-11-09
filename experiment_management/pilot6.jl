### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 237a05f6-9e0e-11ef-2433-3bdaa51dbed4
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 0d120e19-28c2-4a98-b873-366615a5f784
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	
	set_theme!(th)
end

# ╔═╡ d5811081-d5e2-4a6e-9fc9-9d70332cb338
md"""## Participant management"""

# ╔═╡ 36b348cc-a3bf-41e7-aac9-1f6d858304a2
begin
	# Load data
	PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data,
		reversal_data, jspsych_data = load_pilot6_data(; force_download = true)
	nothing
end

# ╔═╡ cb4f46a2-1e9b-4006-8893-6fc609bcdf52
md""" ## Sanity checks"""

# ╔═╡ 5d487d8d-d494-45a7-af32-7494f1fb70f2
md""" ### PILT"""

# ╔═╡ 2ff04c44-5f86-4617-9a13-6d4228dff359
let
	@assert sort(unique(PILT_data.response)) == sort(["right", "left", "noresp"]) "Unexpected values in response"
	
	@assert all(PILT_data.chosen_feedback .== ifelse.(
		PILT_data.response .== "right",
		PILT_data.feedback_right,
		ifelse.(
			PILT_data.response .== "left",
			PILT_data.feedback_left,
			minimum.(hcat.(PILT_data.feedback_left, PILT_data.feedback_right))
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ d0a2ba1e-8413-48f8-8bbc-542f3555a296
let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp)
end

# ╔═╡ 2897a681-e8dd-4091-a2a0-bd3d4cd23209
md"""### Post-PILT test"""

# ╔═╡ 176c54de-e84c-45e5-872e-2471e575776d
let
	# Select post-PILT test
	test_data_clean = filter(x -> isa(x.block, Int64), test_data)

	@assert unique(test_data_clean.response) == 
	["ArrowRight", "ArrowLeft", nothing] "Unexpected values in respones"

	# Remove missing values
	filter!(x -> !isnothing(x.response), test_data_clean)

	# Create magnitude high and low varaibles
	test_data_clean.magnitude_high = maximum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	test_data_clean.magnitude_low = minimum.(eachrow((hcat(
		test_data_clean.magnitude_left, test_data_clean.magnitude_right))))

	# Create high_chosen variable
	test_data_clean.high_chosen = ifelse.(
		test_data_clean.right_chosen,
		test_data_clean.magnitude_right .== test_data_clean.magnitude_high,
		test_data_clean.magnitude_left .== test_data_clean.magnitude_high
	)

	high_chosen_sum = combine(
		groupby(test_data_clean, :prolific_pid),
		:high_chosen => mean => :acc
	)

	@info "Proportion high magnitude chosen: 
		$(round(mean(high_chosen_sum.acc), digits = 2)), SE=$(round(sem(high_chosen_sum.acc), digits = 2))"

	# Summarize by participant and magnitude
	test_sum = combine(
		groupby(test_data_clean, [:prolific_pid, :magnitude_low, :magnitude_high]),
		:high_chosen => mean => :acc
	)

	test_sum_sum = combine(
		groupby(test_sum, [:magnitude_low, :magnitude_high]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	sort!(test_sum_sum, [:magnitude_low, :magnitude_high])

	mp = data(test_sum_sum) *
	mapping(
		:magnitude_high => nonnumeric => "High magntidue",
		:acc => "Prop. chosen high",
		:se,
		layout = :magnitude_low => nonnumeric
	) * (visual(Errorbars) + visual(ScatterLines))

	draw(mp)

end

# ╔═╡ 18956db1-4ad1-4881-a1e7-8362cf59f011
md"""### WM"""

# ╔═╡ 18e9fccd-cc0d-4e8f-9e02-9782a03093d7
let
	@assert sort(unique(WM_data.response)) == sort(["right", "middle", "left", "noresp"]) "Unexpected values in response"
	
	@assert all(WM_data.chosen_feedback .== ifelse.(
		WM_data.response .== "right",
		WM_data.feedback_right,
		ifelse.(
			WM_data.response .== "left",
			WM_data.feedback_left,
			ifelse.(
				WM_data.response .== "middle",
				WM_data.feedback_middle,
				minimum.(hcat.(WM_data.feedback_left, WM_data.feedback_right))
			)
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ 17666d61-f5fc-4a8d-9624-9ae79f3de6bb
let
	# Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 10)
	WM_data_clean = filter(x -> x.response != "noresp", WM_data_clean)

	# Sumarrize by participant, trial, n_groups
	acc_curve = combine(
		groupby(WM_data_clean, [:prolific_pid, :trial, :n_groups]),
		:response_optimal => mean => :acc
	)

	# Summarize by trial and n_groups
	acc_curve_sum = combine(
		groupby(acc_curve, [:trial, :n_groups]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	acc_curve_sum.lb = acc_curve_sum.acc .- acc_curve_sum.se
	acc_curve_sum.ub = acc_curve_sum.acc .+ acc_curve_sum.se

	# Sort
	sort!(acc_curve_sum, [:n_groups, :trial])

	# Create figure
	f = Figure(size = (700, 350))

	# Create mapping
	mp1 = (data(acc_curve_sum) * (
		mapping(
		:trial => "Trial #",
		:lb,
		:ub,
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1)
	legend!(f[0,1:2], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)


	# Create appearance variable
	sort!(WM_data_clean, [:prolific_pid, :session, :block, :trial])
	
	transform!(
		groupby(WM_data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
		:trial => (x -> 1:length(x)) => :appearance
	)

	# Summarize by appearance
	app_curve = combine(
		groupby(WM_data_clean, [:prolific_pid, :appearance, :n_groups]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:appearance, :n_groups]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:n_groups, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
		:appearance => "Apperance #",
		:lb,
		:ub,
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Band, alpha = 0.5) +
		mapping(
		:appearance => "Apperance #",
		:acc => "Prop. optimal choice",
		group = :n_groups => nonnumeric => "Set size",
		color = :n_groups => nonnumeric => "Set size"
	) * visual(Lines)))
	
	# Plot
	plt2 = draw!(f[1,2], mp2)

	f
end

# ╔═╡ 1d1d6d79-5807-487f-8b03-efb7d0898ae8
md"""### Reversal"""

# ╔═╡ e902cd57-f724-4c26-9bb5-1d03443fb191
let

	# Clean data
	reversal_data_clean = exclude_reversal_sessions(reversal_data; required_n_trials = 120)

	filter!(x -> !isnothing(x.response_optimal), reversal_data_clean)

	# Number trials leading to reversal
	DataFrames.transform!(
		groupby(reversal_data_clean, [:prolific_pid, :block]),
		:trial => (x -> x .- maximum(x) .- 1) => :trial_pre_reversal
	)

	# Count number of block
	DataFrames.transform!(
		groupby(reversal_data_clean, :prolific_pid),
		:block => (x -> maximum(x)) => :n_blocks
	)


	# Summarize accuracy pre reversal
	sum_pre = combine(
		groupby(
			filter(x -> (x.trial_pre_reversal > -4) && 
				(x.block < x.n_blocks) && (x.trial < 48), reversal_data_clean), 
			[:prolific_pid, :trial_pre_reversal]
		),
		:response_optimal => mean => :acc
	)

	sum_pre = combine(
		groupby(sum_pre, :trial_pre_reversal),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	rename!(sum_pre, :trial_pre_reversal => :trial)

	# Summarize post reversal
	sum_post = combine(
		groupby(
			filter(x -> x.trial < 10, reversal_data_clean),
			[:prolific_pid, :trial]
		),
		:response_optimal => mean => :acc
	)

	sum_post = combine(
		groupby(sum_post, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Concatenate pre and post
	sum_pre_post = vcat(sum_pre, sum_post)

	# Create group variable to break line plot
	sum_pre_post.group = sign.(sum_pre_post.trial)

	# Plot
	mp = data(sum_pre_post) *
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

	f = draw(mp; axis = (; xticks = -3:9))
end

# ╔═╡ 91f6a95c-4f2e-4213-8be5-3ca57861ed15
"""
    extract_debrief_responses(data::DataFrame) -> DataFrame

Extracts and processes debrief responses from the experimental data. It filters for debrief trials, then parses and expands JSON-formatted Likert scale and text responses into separate columns for each question.

# Arguments
- `data::DataFrame`: The raw experimental data containing participants' trial outcomes and responses, including debrief information.

# Returns
- A DataFrame with participants' debrief responses. The debrief Likert and text responses are parsed from JSON and expanded into separate columns.
"""
function extract_debrief_responses(data::DataFrame)
	# Select trials
	debrief = filter(x -> !ismissing(x.trialphase) && 
		occursin(r"(acceptability|debrief)", x.trialphase) &&
		!(occursin("pre", x.trialphase)), data)


	# Select variables
	select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])

	# Long to wide
	debrief = unstack(
		debrief,
		[:prolific_pid, :exp_start_time],
		:trialphase,
		:response
	)
	

	# Parse JSON and make into DataFrame
	expected_keys = dropmissing(debrief)[1, Not([:prolific_pid, :exp_start_time])]
	expected_keys = Dict([c => collect(keys(JSON.parse(expected_keys[c]))) 
		for c in names(expected_keys)])
	
	debrief_colnames = names(debrief[!, Not([:prolific_pid, :exp_start_time])])
	
	# Expand JSON strings with defaults for missing fields
	expanded = [
	    DataFrame([
	        # Parse JSON or use empty Dict if missing
	        let parsed = ismissing(row[col]) ? Dict() : JSON.parse(row[col])
	            # Fill missing keys with a default value (e.g., `missing`)
	            Dict(key => get(parsed, key, missing) for key in expected_keys[col])
	        end
	        for row in eachrow(debrief)
	    ])
	    for col in debrief_colnames
	]
	expanded = hcat(expanded...)

	# hcat together
	return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

# ╔═╡ dc957d66-1219-4a97-be46-c6c5c189c8ba
"""
    summarize_participation(data::DataFrame) -> DataFrame

Summarizes participants' performance in a study based on their trial outcomes and responses, for the purpose of approving and paying bonuses.

This function processes experimental data, extracting key performance metrics such as whether the participant finished the experiment, whether they were kicked out, and their respective bonuses (PILT and vigour). It also computes the number of specific trial types and blocks completed, as well as warnings received. The output is a DataFrame with these aggregated values, merged with debrief responses for each participant.

# Arguments
- `data::DataFrame`: The raw experimental data containing participant performance, trial outcomes, and responses.

# Returns
- A summarized DataFrame with performance metrics for each participant, including bonuses and trial information.
"""
function summarize_participation(data::DataFrame)

	function extract_PILT_bonus(outcome)

		if all(ismissing.(outcome)) # Return missing if participant didn't complete
			return missing
		else # Parse JSON
			bonus = filter(x -> !ismissing(x), unique(outcome))[1]
			bonus = JSON.parse(bonus)[1] 
			return bonus
		end

	end
	
	participants = combine(groupby(data, [:prolific_pid, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		:vigour_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :vigour_bonus,
		[:trial_type, :block, :n_stimuli] => 
			((t, b, n) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:trial_type, :block, :n_stimuli] => 
			((t, b, n) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (n .== 3))) => :n_trial_WM,
		[:block, :trial_type, :n_stimuli] => 
			((x, t, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2)])))) => :n_blocks_PILT,
		[:block, :trial_type, :n_stimuli] => 
			((x, t, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 3)])))) => :n_blocks_WM,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "vigour_trial"))) => :n_trials_vigour,
		:trial_presses => (x -> mean(filter(y -> !ismissing(y), x))) => 
			:vigour_average_presses,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "pit_trial"))) => 
			:n_trials_pit,
		[:trialphase, :block] => 
			((t, b) -> length(unique(b[(.!ismissing.(t)) .&& (t .== "reversal")])) - 1) => :n_reversals,
		[:trialphase, :block] => 
			((t, b) -> length(b[(.!ismissing.(t)) .&& (t .== "reversal")])) => :n_trials_reversals,
		:n_warnings => maximum => :n_warnings
	)

	# Compute totla bonus
	insertcols!(participants, :n_trial_PILT, 
		:total_bonus => ifelse.(
			ismissing.(participants.vigour_bonus),
			fill(0., nrow(participants)),
			participants.vigour_bonus
		) .+ ifelse.(
			ismissing.(participants.PILT_bonus),
			fill(0., nrow(participants)),
			participants.PILT_bonus
		)
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
p_sum = summarize_participation(jspsych_data)

# ╔═╡ 6ca0676f-b107-4cc7-b0d2-32cc345dab0d
for r in eachrow(p_sum)
	if r.total_bonus > 0.
		println(r.prolific_pid, ", ", r.total_bonus)
	end
end

# ╔═╡ Cell order:
# ╠═237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═36b348cc-a3bf-41e7-aac9-1f6d858304a2
# ╠═c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
# ╠═6ca0676f-b107-4cc7-b0d2-32cc345dab0d
# ╟─cb4f46a2-1e9b-4006-8893-6fc609bcdf52
# ╟─5d487d8d-d494-45a7-af32-7494f1fb70f2
# ╠═2ff04c44-5f86-4617-9a13-6d4228dff359
# ╟─d0a2ba1e-8413-48f8-8bbc-542f3555a296
# ╟─2897a681-e8dd-4091-a2a0-bd3d4cd23209
# ╟─176c54de-e84c-45e5-872e-2471e575776d
# ╟─18956db1-4ad1-4881-a1e7-8362cf59f011
# ╠═18e9fccd-cc0d-4e8f-9e02-9782a03093d7
# ╠═17666d61-f5fc-4a8d-9624-9ae79f3de6bb
# ╟─1d1d6d79-5807-487f-8b03-efb7d0898ae8
# ╟─e902cd57-f724-4c26-9bb5-1d03443fb191
# ╠═dc957d66-1219-4a97-be46-c6c5c189c8ba
# ╠═91f6a95c-4f2e-4213-8be5-3ca57861ed15
