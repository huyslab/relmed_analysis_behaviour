### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═╡ show_logs = false
begin
	cd("/home/jovyan/")
	import Pkg
	# activate the shared project environment
	Pkg.activate("relmed_environment")
	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	using Tidier, GLM, MixedModels, PlutoUI, LaTeXStrings, ColorSchemes
	using LogExpFunctions: logistic, logit
	import OpenScienceFramework as OSF
	include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "sample_utils.jl"))
	include(joinpath(pwd(), "osf_utils.jl"))
	include(joinpath(pwd(), "vigour_utils.jl"))
	nothing
end

# ╔═╡ 28b7224d-afb4-4474-b346-7ee353b6d3d3
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ 93bc3812-c620-4a8d-a312-de9fd0e55327
begin
	# Load data
	_, _, raw_vigour_data, _, raw_PIT_data, _,
		_, _ = load_pilot6_data()
	nothing
end

# ╔═╡ 9b7e12d3-2230-4286-8159-108133f4f579
begin
	_, _, _, _, _, raw_control_task_data, raw_control_report_data, jspsych_data = load_pilot9_data(;force_download = false)
	nothing
end

# ╔═╡ 7f4013e1-d625-467b-9c3b-df0e123982da
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

# ╔═╡ c7509343-f823-4215-965c-3bc451253036
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
	
	participants = combine(groupby(data, [:prolific_pid, :session, :record_id, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		[:trial_type, :trialphase, :block, :n_stimuli] => 
			((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:block, :trial_type, :trialphase, :n_stimuli] => 
			((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2) .& (.!ismissing.(p) .&& p .!= "PILT_test")])))) => :n_blocks_PILT,
		# :trialphase => (x -> sum(skipmissing(x .∈ Ref(["control_explore", "control_predict_homebase", "control_reward"])))) => :n_trial_control,
		# :trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
		:n_warnings => maximum => :n_warnings,
		:time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration,
		:total_bonus => (x -> all(ismissing.(x)) ? missing : only(skipmissing(x))) => :bonus
		# :trialphase => (x -> sum(skipmissing(x .== "control_instruction_quiz_failure"), init=0)) => :n_quiz_failure
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ 1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
md"""
Set theme globally
"""

# ╔═╡ 1c0196e9-de9a-4dfa-acb5-357c02821c5d
begin
	spearman_brown(
	r;
	n = 2 # Number of splits
	) = (n * r) / (1 + (n - 1) * r)

	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	inch = 96
	pt = 4/3
	mm = inch / 25.4
	cm = mm / 10

	two_color_palette1 = [
		colorant"#F2467E", # Pink
		colorant"#46B4E0"  # Blue
	]

	two_color_palette2 = [
		colorant"#009ADE", # Blue
		colorant"#FFC61E" # Yellow
	]

	two_color_palette3 = [
		colorant"#FF1F5B", # Red
		colorant"#009ADE"  # Blue
	]

	colors=ColorSchemes.PRGn_7.colors;
	colors[4]=colorant"rgb(210, 210, 210)";
	
	th = merge(theme_minimal(), Theme(
		font = "Helvetica",
		fontsize = 28pt,
		Axis = (
			xticklabelsize = 24pt,
			yticklabelsize = 24pt,
			spinewidth = 0.5pt,
			xlabelpadding = 0,
			ylabelpadding = 0
		)
	))
	set_theme!(th)

	function plot_presses_vs_var(vigour_data::DataFrame; x_var::Union{Symbol, Pair{Symbol, typeof(AlgebraOfGraphics.nonnumeric)}}=:reward_per_press, y_var::Symbol=:trial_presses, grp_var::Union{Symbol,Nothing}=nothing, xlab::Union{String,Missing}=missing, ylab::Union{String,Missing}=missing, grplab::Union{String,Missing}=missing, subtitle::String="", combine::Union{Bool,String}=false)
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
												visual(Lines, alpha=0.1, linewidth=1)

			# Create the plot for the average line
			if grp_var === nothing
					average_plot = data(avg_w_data) *
												average_mapping * (
														visual(Errorbars, whiskerwidth=6, legend = (; show = false)) *
														mapping(:se_y) +
														visual(ScatterLines, linewidth=3, markersize=12pt)) *
												visual(color=:dodgerblue2)
			else
					average_plot = data(avg_w_data) *
												average_mapping * (
														visual(Errorbars, whiskerwidth=6pt, linewidth=2, legend = (; show = false)) *
														mapping(:se_y, color=grp_var => grplab_text) +
														visual(ScatterLines, linewidth=3, markersize=12pt))
			end

			# Combine the plots

			# Set up the axis
			xlab_text = ismissing(xlab) ? uppercasefirst(join(split(string(x_var), r"\P{L}+"), " ")) : xlab
			ylab_text = ismissing(ylab) ? uppercasefirst(join(split(string(y_var), r"\P{L}+"), " ")) : ylab

			fig=Figure(;size=(9inch, 4.5inch))
			axis = (;
					xlabel=xlab_text,
					ylabel=ylab_text,
					subtitle=subtitle
			)
			if combine == "average"
					f = draw!(fig[1,1], average_plot, scales(Color = (; palette = two_color_palette3)); axis=axis)
					legend!(fig[1,1], f; 
							tellheight = false,
							tellwidth = false,
							halign = 0.975,
							valign = 0.025,
							fontsize = 24pt,
							orientation = :horizontal
							)
			else
					# Draw the plot
					fig_patch = fig[1, 1] = GridLayout()
					ax_left = Axis(fig_patch[1, 1], ylabel=ylab_text)
					ax_right = Axis(fig_patch[1, 2])
					Label(fig_patch[2, :], xlab_text)
					draw!(ax_left, individual_plot, scales(Color = (; palette = two_color_palette3)))
					f = draw!(ax_right, average_plot, scales(Color = (; palette = two_color_palette3)))
					legend!(fig_patch[1, 2], f; 
							tellheight = false,
							tellwidth = false,
							halign = 0.975,
							valign = 0.025,
							fontsize = 24pt,
							orientation = :horizontal
							)
					# rowgap!(fig_patch, 5)
			end
			return fig
	end

	function RLDM_reliability_scatter!(
		f::GridPosition;
		df::AbstractDataFrame,
		xlabel::AbstractString,
		ylabel::AbstractString,
		xcol::Symbol = :x,
		ycol::Symbol = :y,
		subtitle::AbstractString = "",
		tickformat::Union{Function, Makie.Automatic} = Makie.automatic,
		correct_r::Bool = true, # Whether to apply Spearman Brown
		markersize::Union{Int64,Float64} = 9
	)	

		# Compute correlation
		r = cor(df[!, xcol], df[!, ycol])
		
		# Spearman-Brown correction
		if correct_r
			r = spearman_brown(r)
		end

		# Text
		r_text = "n = $(nrow(df)),$(correct_r ? " SB" : "") r = $(round(r; digits = 2))"

		# Plot
		mp = data(df) *
				mapping(xcol, ycol) *
				(visual(Scatter; markersize = markersize, alpha = 0.75) + linear()) +
			mapping([0], [1]) *
				visual(ABLines, linestyle = :dash, color = :gray70)
		
		draw!(f, mp; axis=(;
			xlabel = xlabel, 
			ylabel = ylabel,
			xtickformat = tickformat,
			ytickformat = tickformat,
			subtitle = subtitle
		))

		if r > 0
			Label(
				f,
				r_text,
				fontsize = 24pt,
				font = :bold,
				halign = 0.975,
				valign = 0.025,
				tellheight = false,
				tellwidth = false
			)
		end

	end
end

# ╔═╡ e0a27119-57c2-43b7-89c4-70624c0c665d
md"""
# Vigour
"""

# ╔═╡ de48ee97-d79a-46e4-85fb-08dd569bf7ef
begin
	vigour_unfinished = @chain raw_vigour_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	vigour_data = @chain raw_vigour_data begin
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0") # From sess1
		@filter(!(prolific_pid in ["6721ec463c2f6789d5b777b5", "62ae1ecc1bd29fdc6b14f6ea", "672c8f7bd981bf863dd16a98"])) # From sess2
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true)
		)
		@anti_join(vigour_unfinished)
	end
	nothing
end

# ╔═╡ bd55dd69-c927-45e2-98cf-04f0aa919853
md"""
## Press rate by reward rate
"""

# ╔═╡ d970091a-9316-4d9f-b7ba-9ac0eaf36ae4
let
	two_sess_sub = combine(groupby(vigour_data, :prolific_pid), :session => length∘unique => :n_session) |>
	x -> filter(:n_session => (==(2)), x)
	fig = plot_presses_vs_var(@filter(vigour_data, trial_number > 1); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:session, xlab="Reward/press", ylab = "Press/sec", combine=false)

	# Save
	filepaths = joinpath("results/RLDM/poster", "Vigour_press_by_reward_rate_session")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end


# ╔═╡ 18f08be5-ffbe-455a-a870-57df5c007e01
md"""
## Test-retest Reliability
"""

# ╔═╡ 33a27773-b242-49b3-9318-59c15e9602f9
let
	motor_retest_df = @chain vigour_data begin
		@group_by(prolific_pid, session)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = session, values_from = n_presses)
		@drop_missing
	end

	sensitivity_retest_df = @chain vigour_data begin
		@filter(trial_number != 0)
		@arrange(prolific_pid, session, reward_per_press)
		@group_by(prolific_pid, session)
		@mutate(low_rpp = if_else(reward_per_press <= median(reward_per_press), "low_rpp", "high_rpp"))
		@ungroup
		@group_by(prolific_pid, session, low_rpp)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = low_rpp, values_from = n_presses)
		@mutate(low_to_high_diff = -(low_rpp - high_rpp))
		@select(-ends_with("_rpp"))
		@pivot_wider(names_from = session, values_from = low_to_high_diff)
		@drop_missing
	end
	
	fig=Figure(;size=(9inch, 4.5inch))
	RLDM_reliability_scatter!(
		fig[1,1];
		df=motor_retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Avg. Press Rate",
		correct_r=false
	)

	RLDM_reliability_scatter!(
		fig[1,2];
		df=sensitivity_retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Reward sensitivity",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/RLDM/poster", "Vigour_retest")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end


# ╔═╡ 6e7c866b-b76d-42ee-836d-e63597c71dea
md"""
# PIT
"""

# ╔═╡ c5e01e59-ce51-4e16-a940-8fbfdd907f5b
begin
	raw_PIT_data.coin_name = recode(raw_PIT_data.coin, -1.0 => "-£1", -0.5 => "-50p", -0.01 => "-1p", 0.0 => "0", 0.01 => "1p", 0.5 => "50p", 1.0 => "£1")
	PIT_unfinished = @chain raw_PIT_data begin
		@count(prolific_pid, exp_start_time)
		@mutate(most_common_n = ~mode(n))
		@filter(n < most_common_n)
	end
	PIT_data = @chain raw_PIT_data begin
		@filter(prolific_pid != "671139a20b977d78ec2ac1e0") # From sess1
		@filter(!(prolific_pid in ["6721ec463c2f6789d5b777b5", "62ae1ecc1bd29fdc6b14f6ea", "672c8f7bd981bf863dd16a98"])) # From sess2
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true),
			coin_cat = categorical(coin_name; levels = ["-£1", "-50p", "-1p", "0", "1p", "50p", "£1"], ordered=true)
		)
		@anti_join(PIT_unfinished)
	end
	nothing;
end

# ╔═╡ 65e3d59b-be9f-4d07-b986-94e09bda5f87
let
	grouped_data, avg_w_data = avg_presses_w_fn(PIT_data, [:coin_cat], :press_per_sec, :session)
	p = data(avg_w_data) *
	mapping(:coin_cat, :avg_y, col=:session) *
	(
		visual(Lines, linewidth=3, color=:gray75) +
		visual(Errorbars, whiskerwidth=6pt, linewidth = 2) *
		mapping(:se_y, color=:coin_cat => :"Coin value") +
		visual(Scatter, markersize=12pt) *
		mapping(color=:coin_cat => :"Coin value")
	)
    fig = Figure(size = (9inch, 4.5inch))
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors), Col = (; categories = ["1" => "Session 1", "2" => "Session 2"])); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelsize=20pt))

	# Save
	filepaths = joinpath("results/RLDM/poster", "PIT_press_by_pavlovian")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end

# ╔═╡ 4fa24da6-8e94-4773-8311-d1b0b395cacc
let
	val_diff_retest_df = @chain PIT_data begin
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid], :session, :diff)
		dropmissing()
	end

	val_asymm_retest_df = @chain PIT_data begin
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = zero - (pos + neg)/2)
		unstack([:prolific_pid], :session, :asymm)
		dropmissing()
	end
	
	fig=Figure(;size=(9inch, 4.5inch))
	RLDM_reliability_scatter!(
		fig[1,1];
		df=val_diff_retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="PIT effect",
		correct_r=false
	)

	RLDM_reliability_scatter!(
		fig[1,2];
		df=val_asymm_retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Valence asymmetry",
		correct_r=false
	)
	
	# Save
	filepaths = joinpath("results/RLDM/poster", "PIT_retest")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end


# ╔═╡ 78419cbf-b1e6-41f5-8113-a7c320f20e16
md"""
# Control
"""

# ╔═╡ 4964ddc9-0139-4a40-9daa-89878b10ba96
begin
	p_sum = summarize_participation(jspsych_data)
	p_no_double_take = exclude_double_takers(p_sum) |>
	x -> filter(x -> !ismissing(x.finished) && x.finished, x)
	p_finished = filter(x -> !ismissing(x.finished) && x.finished, p_sum)
	control_task_data = semijoin(raw_control_task_data, p_no_double_take, on=:record_id)
	control_report_data = semijoin(raw_control_report_data, p_no_double_take, on=:record_id)
	@assert all(combine(groupby(control_task_data, :record_id), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created *incorrectly* in chronological order"
	nothing;
end

# ╔═╡ 831ab45f-d7c1-4e6e-a7f1-07acf2adb8cc
let
	threshold_df = (; y = [6, 12, 18], threshold = ["Low", "Mid", "High"])
	spec2 = data(threshold_df) * mapping(:y) * visual(HLines, linewidth=2, linestyle=:dash, color=:gray70)
	
	grouped_data, avg_w_data = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		avg_presses_w_fn([:current, :session], :trial_presses)
	end
	transform!(avg_w_data, 
		[:avg_y, :se_y] => ((x, y) -> x + y) => :upper,
		[:avg_y, :se_y] => ((x, y) -> x - y) => :lower
	)
	
	ind_p = data(grouped_data) * (
			mapping(:current => nonnumeric, :mean_y, color=:session => "Session", group=:prolific_pid, col=:session) * visual(Lines, linewidth=1, alpha = 0.2, legend = (; show = false))
		)

	avg_p = data(avg_w_data) * mapping(:current => nonnumeric, :avg_y, color=:session => "Session", col=:session) * (visual(ScatterLines, linewidth=3, markersize=12pt, legend = (; show = false)) + mapping(:se_y, col=:session) * visual(Errorbars, whiskerwidth=6pt, linewidth=2, legend = (; show = false)))

	all_p = spec2 + ind_p + avg_p

	fig=Figure(;size=(9inch, 4.5inch))
	f = draw!(fig[1,1], all_p, scales(Color = (; palette = two_color_palette3), Col = (;categories=["1" => "Session 1", "2" => "Session 2"])); axis=(; yticks=0:6:24, xlabel = "Current strength", ylabel = "Key presses"))

	# Save
	filepaths = joinpath("results/RLDM/poster", "Control_explore_presses")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end

# ╔═╡ a00197b4-dde7-413f-acfd-7a6a9dc7bd93
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	retest_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@group_by(session)
		@mutate(current = current - mean(current))
		@ungroup()
 		groupby([:prolific_pid, :session])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(9inch, 4.5inch))
	RLDM_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle=L"\beta_0",
		correct_r=false
	)
	RLDM_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_current)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle=L"β_{Current}",
		correct_r=false
	)
	Label(fig[0,:], L"\text{Presses} \sim β_0 + β_1\times \text{Current}", fontsize = 24pt, font=:bold)

	# Save
	filepaths = joinpath("results/RLDM/poster", "Control_retest")
	save(filepaths * ".png", fig; px_per_unit = 150/inch)
	save(filepaths * ".pdf", fig; px_per_unit = 150/inch)

	fig
end

# ╔═╡ Cell order:
# ╠═b41e7252-a075-11ef-039c-f532a7fb0a94
# ╠═28b7224d-afb4-4474-b346-7ee353b6d3d3
# ╟─93bc3812-c620-4a8d-a312-de9fd0e55327
# ╟─9b7e12d3-2230-4286-8159-108133f4f579
# ╟─c7509343-f823-4215-965c-3bc451253036
# ╟─7f4013e1-d625-467b-9c3b-df0e123982da
# ╟─1f8ca836-f2f7-4965-bf07-5656cf6c4ec6
# ╠═1c0196e9-de9a-4dfa-acb5-357c02821c5d
# ╟─e0a27119-57c2-43b7-89c4-70624c0c665d
# ╟─de48ee97-d79a-46e4-85fb-08dd569bf7ef
# ╟─bd55dd69-c927-45e2-98cf-04f0aa919853
# ╠═d970091a-9316-4d9f-b7ba-9ac0eaf36ae4
# ╟─18f08be5-ffbe-455a-a870-57df5c007e01
# ╟─33a27773-b242-49b3-9318-59c15e9602f9
# ╟─6e7c866b-b76d-42ee-836d-e63597c71dea
# ╟─c5e01e59-ce51-4e16-a940-8fbfdd907f5b
# ╟─65e3d59b-be9f-4d07-b986-94e09bda5f87
# ╠═4fa24da6-8e94-4773-8311-d1b0b395cacc
# ╟─78419cbf-b1e6-41f5-8113-a7c320f20e16
# ╠═4964ddc9-0139-4a40-9daa-89878b10ba96
# ╠═831ab45f-d7c1-4e6e-a7f1-07acf2adb8cc
# ╠═a00197b4-dde7-413f-acfd-7a6a9dc7bd93
