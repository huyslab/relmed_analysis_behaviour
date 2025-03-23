### A Pluto.jl notebook ###
# v0.20.3

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
	using Tidier
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

# ╔═╡ e60b5430-adef-4f12-906c-9b70de436833
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data, max_press_data, jspsych_data = load_pilot7_data(; force_download = false, return_version = "7.0")
end
  ╠═╡ =#

# ╔═╡ cb4f46a2-1e9b-4006-8893-6fc609bcdf52
md""" ## Sanity checks"""

# ╔═╡ 5d487d8d-d494-45a7-af32-7494f1fb70f2
md""" ### PILT"""

# ╔═╡ 2ff04c44-5f86-4617-9a13-6d4228dff359
#=╠═╡
let
	@assert all((x -> x in ["right", "left", "noresp"]).(unique(PILT_data.response))) "Unexpected values in response"
	
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
  ╠═╡ =#

# ╔═╡ d0a2ba1e-8413-48f8-8bbc-542f3555a296
#=╠═╡
let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
	# PILT_data_clean = filter(x -> x.block <= 5, PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

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
	
	
	draw(mp; legend = (; show = false))
end
  ╠═╡ =#

# ╔═╡ 13b4b55e-ad52-4c05-b9ec-9eb802cffae0
#=╠═╡
let
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 20)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)
end
  ╠═╡ =#

# ╔═╡ 2897a681-e8dd-4091-a2a0-bd3d4cd23209
md"""### Post learning test phases"""

# ╔═╡ 18956db1-4ad1-4881-a1e7-8362cf59f011
md"""### WM"""

# ╔═╡ 18e9fccd-cc0d-4e8f-9e02-9782a03093d7
#=╠═╡
let
	@assert all((x -> x in ["right", "middle", "left", "noresp"]).(unique(WM_data.response))) "Unexpected values in response"
	
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
  ╠═╡ =#

# ╔═╡ 17666d61-f5fc-4a8d-9624-9ae79f3de6bb
#=╠═╡
let
	# Clean data
	WM_data_clean = exclude_PLT_sessions(WM_data, required_n_blocks = 1)
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
  ╠═╡ =#

# ╔═╡ ce1f3229-1675-495a-9a8a-5ae94194bdce
md"""
### Max press rate
"""

# ╔═╡ eabc7639-612f-4bad-843e-1d8e802740c7
#=╠═╡
filter!(x -> x.trial_presses > 0, max_press_data)
  ╠═╡ =#

# ╔═╡ 0ca7bef1-439c-4718-b620-9dd8a0cc35fd
#=╠═╡
@chain max_press_data begin
	data(_) * mapping(:avg_speed) * visual(Hist)
	draw
end
  ╠═╡ =#

# ╔═╡ 516f2e8c-3ef4-406e-a2cb-b735b84b9ec4
#=╠═╡
@chain max_press_data begin
	describe(:all)
end
  ╠═╡ =#

# ╔═╡ f242aecf-b5c3-47ed-a511-f0b9e75f209c
#=╠═╡
quantile(max_press_data.avg_speed, [0.1, 0.25, 0.5, 0.75, 0.9])
  ╠═╡ =#

# ╔═╡ 7559e78d-7bd8-4450-a215-d74a0b1d670a
md"""
### Vigour
"""

# ╔═╡ 7563e3f6-8fe2-41cc-8bdf-c05c86e3285e
#=╠═╡
begin
	filter!(x -> !(x.prolific_pid in []), vigour_data);
	transform!(vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	nothing;
end
  ╠═╡ =#

# ╔═╡ 7be9ab7d-357f-4833-9660-d0678fb3a672
#=╠═╡
data(vigour_data) * 
	mapping(:magnitude => nonnumeric, :press_per_sec; col=:ratio => nonnumeric) * 
	visual(RainClouds) |> draw
  ╠═╡ =#

# ╔═╡ 243e92bc-b2fb-4f76-9de3-08f8a2e4b25d
#=╠═╡
begin
	@chain vigour_data begin
		@filter(press_per_sec > 11)
		@count(prolific_pid)
	end
end
  ╠═╡ =#

# ╔═╡ 0312ce5f-be36-4d9b-aee3-04497f846537
#=╠═╡
let
	n_miss_df = @chain vigour_data begin
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in Vigour task"))
end
  ╠═╡ =#

# ╔═╡ 2bd5807d-776d-44af-88f3-29f4eb17a1a0
#=╠═╡
let
	avg_df = @chain vigour_data begin
		@group_by(trial_number)
		@summarize(trial_presses = mean(trial_presses), se = mean(trial_presses)/sqrt(length(prolific_pid)))
		@ungroup
	end
	p = data(vigour_data) * mapping(:trial_number, :trial_presses) * AlgebraOfGraphics.linear() + data(avg_df) * mapping(:trial_number, :trial_presses) * visual(ScatterLines)
    draw(p)
end
  ╠═╡ =#

# ╔═╡ 3d05e879-aa5c-4840-9f4f-ad35b8d9519a
#=╠═╡
let
	test_acc_df = @chain post_vigour_test_data begin
		@mutate(
			diff_rpp = (left_magnitude/left_ratio) - (right_magnitude/right_ratio),
			chose_left = Int(response === "ArrowLeft")
		)
		@group_by(prolific_pid, version)
		@select(diff_rpp, chose_left)
		@mutate(chose_left = chose_left * 2 - 1, truth = sign(diff_rpp))
		@mutate(acc = chose_left == truth)
		@summarize(acc = mean(acc))
		@ungroup
	end
	@info "Vigour acc: $(round(mean(test_acc_df.acc); digits=2))"
	data(test_acc_df) *
	mapping(:acc) *
	visual(Hist) |>
	draw(;axis=(;xlabel="Accuracy",ylabel="Count (#Participant)"))
end
  ╠═╡ =#

# ╔═╡ 665aa690-4f37-4a31-b87e-3b4aee66b3b1
md"""
### PIT
"""

# ╔═╡ 43d5b727-9761-48e3-bbc6-89af0c4f3116
#=╠═╡
begin
	filter!(x -> !(x.prolific_pid in []), PIT_data);
	transform!(PIT_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	nothing;
end
  ╠═╡ =#

# ╔═╡ 89258a40-d4c6-4831-8cf3-d69d984c4f6e
#=╠═╡
let
	n_miss_df =  @chain PIT_data begin
		# @filter(coin != 0)
		@arrange(prolific_pid, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@group_by(prolific_pid, pig)
		@summarize(n_miss = sum(trial_presses == 0))
		@ungroup()
		@mutate(short_id = last.(prolific_pid, 5))
	end
	# Create the plot
	plot = data(n_miss_df) * 
		   mapping(:n_miss, layout=:pig) * 
		   histogram(bins=5)
	
	# Set up the axis
	axis = (
		xlabel = "# No-response trials",
		ylabel = "Count (# participants)"
	)
	
	# Draw the plot
	draw(plot; axis, figure=(;title="No-response trial distribution in PIT task"))
end
  ╠═╡ =#

# ╔═╡ ffd08086-f12c-4b8a-afb6-435c8729241e
#=╠═╡
let
	PIT_acc_df = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			magnitude_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => magnitude_left),
			magnitude_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => magnitude_right)
		)
		@group_by(same_valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.same_valence.==false][1]; digits=2))"
	@info "PIT acc for in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.same_valence.==true][1]; digits=2))"
	@chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			magnitude_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => magnitude_left),
			magnitude_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => magnitude_right)
		)
		@group_by(prolific_pid, exp_start_time, session, same_valence)
		@summarize(acc = mean(skipmissing((magnitude_right > magnitude_left) == right_chosen)))
		@ungroup
		data(_) * mapping(:same_valence => nonnumeric => "Same valence", :acc => "PIT test accuracy", color=:same_valence => nonnumeric => "Same valence") * visual(RainClouds)
		draw()
	end
end
  ╠═╡ =#

# ╔═╡ b3fdfcd9-ca07-4433-b5bd-fe660cc8c0db
#=╠═╡
let
	avg_df = @chain PIT_data begin
		@filter(coin==0)
		@group_by(trial_number)
		@summarize(trial_presses = mean(trial_presses), se = mean(trial_presses)/sqrt(length(prolific_pid)))
		@ungroup
	end
	p = data(@filter(PIT_data, coin==0)) * mapping(:trial_number, :trial_presses) * AlgebraOfGraphics.linear() + data(avg_df) * mapping(:trial_number, :trial_presses) * visual(ScatterLines)
    draw(p)
end
  ╠═╡ =#

# ╔═╡ 9d159f32-4f76-439b-8226-909d1f8ff347
#=╠═╡
@chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			magnitude_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => magnitude_left),
			magnitude_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => magnitude_right)
		)
		@filter(same_valence)
end
  ╠═╡ =#

# ╔═╡ 29d320df-c984-496a-8d81-6967dd72e964
#=╠═╡
jspsych_data |> names
  ╠═╡ =#

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
			return maximum([0., bonus])
		end

	end
	
	participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		:vigour_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :vigour_bonus,
		[:trial_type, :trialphase, :block, :n_stimuli] => 
			((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:trial_type, :block, :n_stimuli] => 
			((t, b, n) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (n .== 3))) => :n_trial_WM,
		[:block, :trial_type, :trialphase, :n_stimuli] => 
			((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2) .& (.!ismissing.(p) .&& p .!= "PILT_test")])))) => :n_blocks_PILT,
		[:block, :trial_type, :n_stimuli] => 
			((x, t, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 3)])))) => :n_blocks_WM,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "vigour_trial"))) => :n_trials_vigour,
		:trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
		:trial_presses => (x -> mean(filter(y -> !ismissing(y), x))) => 
			:vigour_average_presses,
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "pit_trial"))) => 
			:n_trials_pit,
		:n_warnings => maximum => :n_warnings,
		:time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration
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
#=╠═╡
begin
	p_sum = summarize_participation(jspsych_data)
	@info "# Valid data samples: $(sum(skipmissing(p_sum.finished)))"
end
  ╠═╡ =#

# ╔═╡ eeffa44e-a9e6-43dd-b47d-00670299e0f2
#=╠═╡
let

	for r in eachrow(filter(x -> !ismissing(x.finished), p_sum))
		println("$(r.prolific_pid), $(round(r.total_bonus, digits = 2))")
	end

	
	p_sum
end
  ╠═╡ =#

# ╔═╡ ce27b319-d728-46f5-aaf1-051fe252bf8b
function avg_presses_w_fn(vigour_data::DataFrame, x_var::Vector{Symbol}, y_var::Symbol, grp_var::Union{Symbol,Nothing}=nothing)
    # Define grouping columns
    group_cols = grp_var === nothing ? [:prolific_pid, x_var...] : [:prolific_pid, grp_var, x_var...]
    # Group and calculate mean presses for each participant
    grouped_data = groupby(vigour_data, Cols(group_cols...)) |>
                   x -> combine(x, y_var => mean => :mean_y) |>
                        x -> sort(x, Cols(group_cols...))
    # Calculate the average across all participants
    avg_w_data = @chain grouped_data begin
        @group_by(prolific_pid)
        @mutate(sub_mean = mean(mean_y))
        @ungroup
        @mutate(grand_mean = mean(mean_y))
        @mutate(mean_y_w = mean_y - sub_mean + grand_mean)
        groupby(Cols(grp_var === nothing ? x_var : [grp_var, x_var...]))
        @summarize(
            n = n(),
            avg_y = mean(mean_y),
            se_y = std(mean_y_w) / sqrt(length(prolific_pid)))
        @ungroup
    end
    return grouped_data, avg_w_data
end

# ╔═╡ 9ad3f111-7b4b-45c5-bc9d-cce3bd0e0c72
#=╠═╡
let
	df = @chain PIT_data begin
		@arrange(prolific_pid, session, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:coin, :pig], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig=>nonnumeric) *
	(
	visual(Lines, linewidth=1, color=:gray) +
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y, color=:coin => nonnumeric) +
	visual(Scatter) *
	mapping(color=:coin => nonnumeric)
	)
	draw(p, scales(Color = (; palette=:PRGn_7)); axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150, xticklabelrotation=pi/4))
end
  ╠═╡ =#

# ╔═╡ 8f6d8e98-6d73-4913-a02d-97525176549a
#=╠═╡
let
	df = @chain PIT_data begin
		@mutate(session=if_else(trial_number <= 36, "Trial: 1-36", "Trial: 37-72"))
		@arrange(prolific_pid, session, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:session, :coin, :pig], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig=>nonnumeric, row=:session=>nonnumeric) *
	(
	visual(Lines, linewidth=1, color=:gray) +
	visual(Errorbars, whiskerwidth=4) *
	mapping(:se_y, color=:coin => nonnumeric) +
	visual(Scatter) *
	mapping(color=:coin => nonnumeric)
	)
	draw(p, scales(Color = (; palette=:PRGn_7)); axis=(;xlabel="Pavlovian stimuli (coin)", ylabel="Press/sec", width=150, height=150, xticklabelrotation=pi/4))
end
  ╠═╡ =#

# ╔═╡ e3f88292-fdb9-4628-88ee-8d935f00a761
function plot_presses_vs_var(vigour_data::DataFrame; x_var::Union{Symbol, Pair{Symbol, typeof(AlgebraOfGraphics.nonnumeric)}}=:reward_per_press, y_var::Symbol=:trial_presses, grp_var::Union{Symbol,Nothing}=nothing, xlab::Union{String,Missing}=missing, ylab::Union{String,Missing}=missing, grplab::Union{String,Missing}=missing, combine::Bool=false)
    plain_x_var = isa(x_var, Pair) ? x_var.first : x_var
    grouped_data, avg_w_data = avg_presses_w_fn(vigour_data, [plain_x_var], y_var, grp_var)

	# Set up the legend title
    grplab_text = ismissing(grplab) ? uppercasefirst(join(split(string(grp_var), r"\P{L}+"), " ")) : grplab
	
    # Define mapping based on whether grp_var is provided
    individual_mapping = grp_var === nothing ?
                         mapping(x_var, :mean_y, group=:prolific_pid) :
                         mapping(x_var, :mean_y, color=grp_var => grplab_text, group=:prolific_pid)

    average_mapping = grp_var === nothing ?
                      mapping(x_var, :avg_y) :
                      mapping(x_var, :avg_y, color=grp_var => grplab_text)

    # Create the plot for individual participants
    individual_plot = data(grouped_data) *
                      individual_mapping *
                      visual(Lines, alpha=0.15, linewidth=1)

    # Create the plot for the average line
    if grp_var === nothing
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y) +
                           visual(ScatterLines, linewidth=2)) *
                       visual(color=:dodgerblue)
    else
        average_plot = data(avg_w_data) *
                       average_mapping * (
                           visual(Errorbars, whiskerwidth=4) *
                           mapping(:se_y, color=grp_var => grplab_text) +
                           visual(ScatterLines, linewidth=2))
    end

    # Combine the plots
    fig = Figure(
        size=(12.2, 7.6) .* 144 ./ 2.54, # 144 points per inch, then cm
    )

    # Set up the axis
    xlab_text = ismissing(xlab) ? uppercasefirst(join(split(string(x_var), r"\P{L}+"), " ")) : xlab
    ylab_text = ismissing(ylab) ? uppercasefirst(join(split(string(y_var), r"\P{L}+"), " ")) : ylab

    if combine
        axis = (;
            xlabel=xlab_text,
            ylabel=ylab_text,
        )
        final_plot = individual_plot + average_plot
        fig = draw(final_plot; axis=axis)
    else
        # Draw the plot
        fig_patch = fig[1, 1] = GridLayout()
        ax_left = Axis(fig_patch[1, 1], ylabel=ylab_text)
        ax_right = Axis(fig_patch[1, 2])
        Label(fig_patch[2, :], xlab_text)
        draw!(ax_left, individual_plot)
        f = draw!(ax_right, average_plot)
        legend!(fig_patch[1, 3], f)
        rowgap!(fig_patch, 5)
    end
    return fig
end

# ╔═╡ 814aec54-eb08-4627-9022-19f41bcdac9f
#=╠═╡
let
	plot_presses_vs_var(@filter(vigour_data, trial_number > 0); x_var=:reward_per_press, y_var=:press_per_sec, xlab="Reward/press", ylab = "Press/sec", combine=false)
end
  ╠═╡ =#

# ╔═╡ a6794b95-fe5e-4010-b08b-f124bff94f9f
#=╠═╡
let
	common_rpp = unique(PIT_data.reward_per_press)
	instrumental_data = @chain PIT_data begin
		@filter(coin==0)
		@bind_rows(vigour_data)
		@mutate(trialphase=categorical(trialphase, levels=["vigour_trial", "pit_trial"], ordered=true))
		# @mutate(trialphase=~recode(trialphase, "vigour_trial" => "Vigour", "pit_trial" => "PIT w/o coin"))
		@filter(reward_per_press in !!common_rpp)
	end
	plot_presses_vs_var(@bind_rows(PIT_data, instrumental_data); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:trialphase, xlab="Reward/press", ylab = "Press/sec", combine=false)
end
  ╠═╡ =#

# ╔═╡ cd627424-5926-4c99-a515-1dc320d49c65
function sanity_check_test(test_data_clean::DataFrame)
		@assert Set(test_data_clean.response) in 
		[Set(["right", "left", "noresp"]), Set(["right", "left"])] "Unexpected values in respones: $(unique(test_data_clean.response))"
	
		# Remove missing values
		filter!(x -> !(x.response .== "noresp"), test_data_clean)
	
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
			groupby(filter(x -> x.magnitude_high != x.magnitude_low, test_data_clean), :prolific_pid),
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

		if length(unique(test_data_clean.prolific_pid)) > 1
			mp = mapping(
				:magnitude_high => nonnumeric => "High magntidue",
				:acc => "Prop. chosen high",
				:se,
				layout = :magnitude_low => nonnumeric
			) * (visual(Errorbars) + visual(ScatterLines))
		else
			mp = mapping(
				:magnitude_high => nonnumeric => "High magntidue",
				:acc => "Prop. chosen high",
				layout = :magnitude_low => nonnumeric
			) * (visual(Scatter) + visual(ScatterLines))
		end
				
		mp = data(test_sum_sum) *
			mp
	
		draw(mp)
	end

# ╔═╡ 176c54de-e84c-45e5-872e-2471e575776d
#=╠═╡
let
	# Select post-PILT test
	test_data_clean = filter(x -> isa(x.block, Int64) && (x.block < 6), test_data)

	sanity_check_test(test_data_clean)

end
  ╠═╡ =#

# ╔═╡ 9405f724-0fd7-40fe-85bc-bef22707e6fa
#=╠═╡
let
	# Select post-WM test
	test_data_clean = filter(x -> isa(x.block, Int64) && (x.block == 6), test_data)

	sanity_check_test(test_data_clean)

end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═e60b5430-adef-4f12-906c-9b70de436833
# ╠═c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
# ╠═eeffa44e-a9e6-43dd-b47d-00670299e0f2
# ╟─cb4f46a2-1e9b-4006-8893-6fc609bcdf52
# ╟─5d487d8d-d494-45a7-af32-7494f1fb70f2
# ╠═2ff04c44-5f86-4617-9a13-6d4228dff359
# ╠═d0a2ba1e-8413-48f8-8bbc-542f3555a296
# ╠═13b4b55e-ad52-4c05-b9ec-9eb802cffae0
# ╠═2897a681-e8dd-4091-a2a0-bd3d4cd23209
# ╠═176c54de-e84c-45e5-872e-2471e575776d
# ╠═9405f724-0fd7-40fe-85bc-bef22707e6fa
# ╟─18956db1-4ad1-4881-a1e7-8362cf59f011
# ╠═18e9fccd-cc0d-4e8f-9e02-9782a03093d7
# ╠═17666d61-f5fc-4a8d-9624-9ae79f3de6bb
# ╟─ce1f3229-1675-495a-9a8a-5ae94194bdce
# ╠═eabc7639-612f-4bad-843e-1d8e802740c7
# ╠═0ca7bef1-439c-4718-b620-9dd8a0cc35fd
# ╠═516f2e8c-3ef4-406e-a2cb-b735b84b9ec4
# ╠═f242aecf-b5c3-47ed-a511-f0b9e75f209c
# ╟─7559e78d-7bd8-4450-a215-d74a0b1d670a
# ╟─7563e3f6-8fe2-41cc-8bdf-c05c86e3285e
# ╠═7be9ab7d-357f-4833-9660-d0678fb3a672
# ╟─243e92bc-b2fb-4f76-9de3-08f8a2e4b25d
# ╟─0312ce5f-be36-4d9b-aee3-04497f846537
# ╟─814aec54-eb08-4627-9022-19f41bcdac9f
# ╠═2bd5807d-776d-44af-88f3-29f4eb17a1a0
# ╟─3d05e879-aa5c-4840-9f4f-ad35b8d9519a
# ╟─665aa690-4f37-4a31-b87e-3b4aee66b3b1
# ╟─43d5b727-9761-48e3-bbc6-89af0c4f3116
# ╟─89258a40-d4c6-4831-8cf3-d69d984c4f6e
# ╠═a6794b95-fe5e-4010-b08b-f124bff94f9f
# ╠═9ad3f111-7b4b-45c5-bc9d-cce3bd0e0c72
# ╠═8f6d8e98-6d73-4913-a02d-97525176549a
# ╠═ffd08086-f12c-4b8a-afb6-435c8729241e
# ╠═b3fdfcd9-ca07-4433-b5bd-fe660cc8c0db
# ╠═9d159f32-4f76-439b-8226-909d1f8ff347
# ╠═dc957d66-1219-4a97-be46-c6c5c189c8ba
# ╠═29d320df-c984-496a-8d81-6967dd72e964
# ╟─91f6a95c-4f2e-4213-8be5-3ca57861ed15
# ╟─ce27b319-d728-46f5-aaf1-051fe252bf8b
# ╟─e3f88292-fdb9-4628-88ee-8d935f00a761
# ╠═cd627424-5926-4c99-a515-1dc320d49c65
