### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 656eb242-31f0-11f0-3950-5b952c0f4e01
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
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	nothing
end

# ╔═╡ 7b2bbc61-acc2-49d4-9195-4c60cfcb894a
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

# ╔═╡ c733e42b-92dd-4615-860d-78b1efe22d41
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

# ╔═╡ 32ebb683-0618-49b5-9eea-acd005d8f336
md"""
# Basic task info
"""

# ╔═╡ 443091f5-f46b-4b7c-b7c8-b98f6d9ceba9
begin
	_, raw_test_data, raw_vigour_data, raw_PIT_data, raw_max_press_data, raw_control_task_data, raw_control_report_data, jspsych_data = load_pilot9_data(;force_download = false)
	nothing
end

# ╔═╡ 9ebce139-53fd-43de-bb82-feaf1c759910
PIT_doubt_PID = ["677afec8954cc9493035dc74", "67f7f3a41fcb35f0eaac48ef", "680d70786b6a8ca0af164c89", "66226d86cdae406d48c9e4c6", "681106ac93d01f1615c6f003"]

# ╔═╡ cae69da2-ec7f-4df6-a69d-d0677185d108
md"""
# Vigour
"""

# ╔═╡ d77efaa6-3b94-4947-b474-97d77719d80f
md"""
## Reliability: motor
"""

# ╔═╡ bfd770f5-3142-4a73-b39f-012d1bdb5de4
md"""
## Reliability: Reward rate sensitivity
"""

# ╔═╡ 5230eada-33ba-413f-88a0-96f76164bc12
md"""
# PIT
"""

# ╔═╡ ad6c8a9d-f0a7-4b6f-b012-dbe33b894a12
begin
	colors=ColorSchemes.PRGn_7.colors;
	colors[4]=colorant"rgb(210, 210, 210)";
	nothing;
end

# ╔═╡ d889a3bb-04cc-433f-94e1-a8c71739f82f
md"""
## Reliability: PIT effect (valence difference)
"""

# ╔═╡ b54f2cb1-5dbb-4ebb-9073-78e1209cda15
md"""
$\text{Asymmetry} = \text{Press rate} | \text{Empty} - \frac{\text{Press rate} | \text{Positive} + \text{Press rate} | \text{Negative}}{2}$
"""

# ╔═╡ abe9466b-e4cc-4803-afdf-ada8f482a732
md"""
Test-retest on the first half
"""

# ╔═╡ 6db5310b-7e8b-42e9-bf8a-38c862e9396c
md"""
# Control
"""

# ╔═╡ 5c38974b-06a9-4497-8ed8-0cf30e313abd
md"""
## Control timing
"""

# ╔═╡ cc52da42-9561-4619-8d6a-237f9ac629e8
md"""
## Exploration
"""

# ╔═╡ a395432f-0dba-4a06-bbe2-d00c596c455e
md"""
### Effort
"""

# ╔═╡ db0ef21a-1b00-4eee-983b-f3f553730f14
md"""
### Exploration
"""

# ╔═╡ fb6d4fdf-ec68-46a2-b5dc-07f0e7cc9a40
md"""
## Prediction
"""

# ╔═╡ 188a9121-a69a-4e20-8600-1a8d7ca7040b
md"""
## Reward
"""

# ╔═╡ f74eea14-748b-4b3c-b887-bc9485ecfea3
md"""
# Misc.
"""

# ╔═╡ 8f2909c4-7746-4994-b97e-6acc87aa84a1
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

# ╔═╡ 8312e274-4cf4-4ad2-906d-e357ef5d4755
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

# ╔═╡ 5a136247-6e4b-4f6a-b76c-ba380f18d68c
begin
	p_sum = summarize_participation(jspsych_data)
end

# ╔═╡ 47f6b0b5-ea48-4d47-929e-388991fd52a4
begin
	p_no_double_take = exclude_double_takers(p_sum) |>
		x -> filter(x -> !ismissing(x.finished) && x.finished, x)
	p_finished = filter(x -> !ismissing(x.finished) && x.finished, p_sum)
end

# ╔═╡ b019dcd5-8eaf-4759-83ed-cc343718fd17
@count(p_no_double_take, session)

# ╔═╡ a590ab5d-647a-4d4f-b3d2-9ef03e7b7f73
begin
	filter!(x -> !(x.prolific_pid in []), raw_vigour_data);
	transform!(raw_vigour_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	vigour_data = semijoin(raw_vigour_data, p_no_double_take, on=:record_id)
	nothing;
end

# ╔═╡ 8a0f53d3-2054-4845-b96d-29399b5aa340
let
	plot_presses_vs_var(@filter(vigour_data, trial_number > 0); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:session, xlab="Reward/press", ylab = "Press/sec", combine="average")
end

# ╔═╡ 517492c4-4055-4317-bd45-e0e2fb77991e
let
	two_sess_sub = combine(groupby(vigour_data, :prolific_pid), :session => length∘unique => :n_session) |>
	x -> filter(:n_session => (==(2)), x)
	fig = plot_presses_vs_var(@filter(semijoin(vigour_data, two_sess_sub, on=:prolific_pid), trial_number > 1); x_var=:reward_per_press, y_var=:press_per_sec, grp_var=:session, xlab="Reward/press", ylab = "Press/sec", combine="average")
end

# ╔═╡ d2892577-9e9c-4c23-8b05-87c00b1962f6
let
	retest_df = @chain vigour_data begin
		@group_by(prolific_pid, session)
		@summarize(n_presses = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = session, values_from = n_presses)
		@drop_missing
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest Press Rate (Press/sec)",
		correct_r=false
	)
	fig
end

# ╔═╡ 272ba203-34cb-4ffc-9b35-4e4d23f1c59d
let
	retest_df = @chain vigour_data begin
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
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Reward rate sensitivity",
		correct_r=false
	)

	fig
end

# ╔═╡ 95bb623c-090d-4338-aa3f-7b31d47d87b2
begin
	filter!(x -> !(x.prolific_pid in []), raw_PIT_data);
	transform!(raw_PIT_data, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec);
	PIT_data = @chain raw_PIT_data begin 
		semijoin(p_no_double_take, on=:record_id)
		@mutate(
			press_per_sec = trial_presses / trial_duration * 1000,
			ratio = categorical(ratio; levels = [1, 8, 16], ordered=true),
			magnitude = categorical(magnitude; levels = [1, 2, 5], ordered=true),
			coin_cat = categorical(coin; levels = [-1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0], ordered=true)
		)
	end
	nothing;
end

# ╔═╡ 12b6b34d-cfc5-48cd-8d50-150c4e388405
let
	grouped_data, avg_w_data = avg_presses_w_fn(PIT_data, [:coin_cat], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin_cat, :avg_y) *
	(
		visual(Lines, linewidth=2, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin_cat => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin_cat => :"Coin value")
	)
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors)); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec"))
	# legend!(fig[1,2], p)

	fig
end

# ╔═╡ 8e186be5-f2a2-4285-a62a-9c021b2dd172
let
	df = @chain PIT_data begin
		@arrange(prolific_pid, session, magnitude, ratio)
		@mutate(pig = "Mag " * string(magnitude) * ", FR " * string(ratio))
		@mutate(pig=categorical(pig,levels=["Mag 2, FR 16","Mag 2, FR 8","Mag 5, FR 8","Mag 1, FR 1"]))
	end
	colors=ColorSchemes.PRGn_7.colors
	colors[4]=colorant"rgb(210, 210, 210)"
	grouped_data, avg_w_data = avg_presses_w_fn(df, [:coin, :pig, :session], :press_per_sec)
	p = data(avg_w_data) *
	mapping(:coin=>nonnumeric, :avg_y, col=:pig=>nonnumeric, row=:session) *
	(
		visual(Lines, linewidth=1, color=:gray75) +
		visual(Errorbars, whiskerwidth=4) *
		mapping(:se_y, color=:coin => nonnumeric => :"Coin value") +
		visual(Scatter, markersize=10) *
		mapping(color=:coin => nonnumeric => :"Coin value")
	)
	fig = Figure(;size=(16, 8) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=colors), Row=(;categories=["1" => "Session 1", "2" => "Session 2"])); axis=(;xlabel="Pavlovian stimuli (coin value)", ylabel="Press/sec", xticklabelrotation=pi/4))
	legend!(fig[1,2], p)

	fig
end

# ╔═╡ c39a9349-ebe3-4b64-9775-a6dd04ac0847
let
	retest_df = @chain PIT_data begin
		@filter(coin != 0)
		@mutate(valence = ifelse(coin > 0, "pos", "neg"))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(diff = pos - neg)
		unstack([:prolific_pid], :session, :diff)
		dropmissing()
		# @filter(prolific_pid != "66350579832c040ee8629b73")
	end
	
	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: PIT Valence Difference",
		correct_r=false
	)

	fig
end

# ╔═╡ bbde0375-f1a9-47e3-b7d0-885a4e0b8548
let
	retest_df = @chain PIT_data begin
		@mutate(valence = ifelse(coin == 0, "zero", ifelse(coin > 0, "pos", "neg")))
		@group_by(prolific_pid, session, valence)
		@summarize(press_per_sec = mean(press_per_sec))
		@ungroup
		@pivot_wider(names_from = valence, values_from = press_per_sec)
		@mutate(asymm = zero - (pos + neg)/2)
		unstack([:prolific_pid], :session, :asymm)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: PIT Valence Asymmetry",
		correct_r=false
	)
		
	fig
end

# ╔═╡ 827751f8-ef30-472b-84f5-631fb212ba14
begin
	test_data = semijoin(raw_test_data, p_no_double_take, on=:record_id)
	PIT_acc_df = @chain test_data begin	
		@filter(block == "pavlovian")
		@mutate(
			EV_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => EV_left),
			EV_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => EV_right)
		)
		@mutate(valence = ifelse(EV_left * EV_right < 0, "Different", ifelse(EV_left > 0, "Positive", "Negative")))
		@group_by(valence)
		@summarize(acc = mean(skipmissing((EV_right > EV_left) == (key == "arrowright"))))
		@ungroup
	end
	@info "PIT acc for NOT in the same valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Different"]...; digits=2))"
	@info "PIT acc for in the positive valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Positive"]...; digits=2))"
	@info "PIT acc for in the negative valence: $(round(PIT_acc_df.acc[PIT_acc_df.valence.=="Negative"]...; digits=2))"

	p = @chain test_data begin
		@filter(block == "pavlovian")
		@mutate(
			EV_left = case_when(
				contains(stimulus_left, "PIT4") => -0.01,
				contains(stimulus_left, "PIT6") => -1.0,
				true => EV_left),
			EV_right = case_when(
				contains(stimulus_right, "PIT4") => -0.01,
				contains(stimulus_right, "PIT6") => -1.0,
				true => EV_right)
		)
		@mutate(valence = ifelse(EV_left * EV_right < 0, "Different", ifelse(EV_left > 0, "Positive", "Negative")))
		@mutate(correct = (EV_right > EV_left) == (key == "arrowright"))
		@filter(!ismissing(correct))
		@group_by(prolific_pid, session, valence)
		@summarize(acc = mean(correct))
		@ungroup
		data(_) * mapping(:valence, :acc, color=:valence, row=:session) * visual(RainClouds; clouds=hist, hist_bins = 20, plot_boxplots = false)
	end
	fig = Figure(;size=(8, 6) .* 144 ./ 2.54)
	p = draw!(fig[1,1], p, scales(Color = (; palette=[colorant"gray", ColorSchemes.Set3_5[[4,5]]...])); axis=(;xlabel="Pavlovian valence", ylabel="PIT test accuracy"))

	fig
end

# ╔═╡ 009447fb-3719-49df-af9d-06f2f26908f3
begin
	control_task_data = semijoin(raw_control_task_data, p_no_double_take, on=:record_id)
	control_report_data = semijoin(raw_control_report_data, p_no_double_take, on=:record_id)
	@assert all(combine(groupby(control_task_data, :record_id), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created *incorrectly* in chronological order"
	nothing;
end

# ╔═╡ 9c4338bb-5e7b-443b-afd4-24e87349af9e
begin
	threshold_df = (; y = [6, 12, 18], threshold = ["Low", "Mid", "High"])
	spec2 = data(threshold_df) * mapping(:y, color = :threshold) * visual(HLines)
	@chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_stage = ifelse(trial <= maximum(trial)/2, "First half", "Second half"))
		@group_by(prolific_pid, session, trial_stage, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, session, trial_stage, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage, row=:session) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage, row=:session, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(spec2 + _; axis=(; xlabel = "Current strength", ylabel = "Presses (in explore trials)"), figure=(;size=(12,12) .* 144 ./ 2.54))
	end
end

# ╔═╡ 2b29cc46-f722-4cf3-b1bf-9bb6310b2457
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	retest_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
 		groupby([:prolific_pid, :session])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β0",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_current)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β[Current]",
		correct_r=false
	)
	Label(fig[0,:], "Presses ~ β0 + β1 * Current strength")
	fig
end

# ╔═╡ a83d1fda-81fe-4305-ab8e-b6c4fbce6749
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	retest_df =  @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_number = ~denserank(trial))
		@mutate(half = if_else(trial_number % 2 === 0, "x", "y"))
		groupby([:prolific_pid, :session, :half])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	figs = []
	for s in unique(vigour_data.session)
		fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
		workshop_reliability_scatter!(
			fig[1,1];
			df=dropmissing(unstack(@filter(retest_df, session == !!s), [:prolific_pid, :session], :half, :β0)),
			xlabel="Even trials",
			ylabel="Odd trials",
			xcol=:x,
			ycol=:y,
			subtitle="β0"
		)
		workshop_reliability_scatter!(
			fig[1,2];
			df=unstack(@filter(retest_df, session == !!s),
						[:prolific_pid, :session], :half, :β_current),
			xlabel="Even trials",
			ylabel="Odd trials",
			xcol=:x,
			ycol=:y,
			subtitle="β[Current]"
		)
		Label(fig[0,:], "Session $(s) Presses ~ β0 + β1 * Current strength")
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 8850fa85-de8c-4328-9df0-dde761852e7a
@chain control_task_data begin
	@filter(trialphase == "control_explore")
	@summary(rt)
end

# ╔═╡ fa6064ce-626b-4309-9183-b129e925058b
begin
	explore_by_times = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@arrange(prolific_pid, session, trial)
		@group_by(prolific_pid, session, left)
		@mutate(left_prev_occur = row_number())
		@ungroup()
		@arrange(prolific_pid, session, trial)
		@group_by(prolific_pid, session, right)
		@mutate(right_prev_occur = row_number())
		@ungroup()
		@arrange(prolific_pid, session, trial)
		@drop_missing(response)
		@filter(left_prev_occur != right_prev_occur)
		@mutate(explorative = ((response == "right") && (left_prev_occur > right_prev_occur)) || ((response == "left") && (left_prev_occur < right_prev_occur)))
	end
	@chain explore_by_times begin
		@group_by(session, trial)
		@summarize(explorative = mean(explorative), upper = mean(explorative) + std(explorative)/sqrt(length(explorative)), lower = mean(explorative) - std(explorative)/sqrt(length(explorative)))
		@ungroup
		@arrange(session, trial)
		data(_) * mapping(:trial, :explorative, color=:session) * (visual(Scatter) + linear())
		draw(;axis=(;xlabel = "Trial", ylabel = "Explorative? (by occurence)"), figure=(;size=(600, 400)))
	end
end

# ╔═╡ dc8445f6-965c-4d57-af9d-136059fe72ec
let
	retest_df = @chain explore_by_times begin
		@group_by(prolific_pid, session)
		@summarize(p_explorative = mean(explorative))
		@ungroup
		unstack([:prolific_pid], :session, :p_explorative)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Exploration by occurence",
		correct_r=false
	)
	fig
end

# ╔═╡ 143f8d33-fabd-4af1-b7ba-69984084a533
let
	retest_df = @chain explore_by_times begin
		@mutate(half = ifelse(~denserank(trial) <= maximum(~denserank(trial))/2, "x", "y"))
		@group_by(prolific_pid, session, half)
		@summarize(p_explorative = mean(explorative))
		@ungroup
		unstack([:prolific_pid, :session], :half, :p_explorative)
		@mutate(explore_change = y - x)
		select(Not(Cols(:x, :y)))
		unstack([:prolific_pid], :session, :explore_change)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Exploration tendency change",
		correct_r=false
	)
	fig
end

# ╔═╡ fa930366-523b-467b-8c3d-5fdd77565064
begin
	explore_by_interval = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@arrange(prolific_pid, session, trial)
		groupby([:prolific_pid, :session, :left])
		transform(:trial => (x -> vcat(0, diff(x))) => :left_prev_occur) # trial number includes those for control prediction
		@ungroup()
		@arrange(prolific_pid, session, trial)
		groupby([:prolific_pid, :session, :right])
		transform(:trial => (x -> vcat(0, diff(x))) => :right_prev_occur) # trial number includes those for control prediction
		@ungroup()
		@arrange(prolific_pid, session, trial)
		@drop_missing(response)
		@filter(left_prev_occur != right_prev_occur)
		@mutate(explorative = ((response == "right") && (left_prev_occur > right_prev_occur)) || ((response == "left") && (left_prev_occur < right_prev_occur)))
	end
	@chain explore_by_interval begin
		@group_by(session, trial)
		@summarize(explorative = mean(explorative), upper = mean(explorative) + std(explorative)/sqrt(length(explorative)), lower = mean(explorative) - std(explorative)/sqrt(length(explorative)))
		@ungroup
		@arrange(session, trial)
		data(_) * mapping(:trial, :explorative, color=:session) * (visual(Scatter) + linear())
		draw(;axis=(;xlabel = "Trial", ylabel = "Explorative? (by interval)"), figure=(;size=(600, 400)))
	end
end

# ╔═╡ 8f1ab7ba-0fa0-4b0f-9b69-9d8440e60f12
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@drop_missing(correct)
		@group_by(session, trial, ship)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		@ungroup
		data(_) * (mapping(:trial, :lower, :upper, color=:ship, col=:session) * visual(Band, alpha = 0.1) + mapping(:trial, :acc, color=:ship, col=:session) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Home base predict acc."), figure=(;size=(800, 400)))
	end
end

# ╔═╡ d0b7207c-4e58-48ea-af3d-2503134b3f5b
let
	retest_df = @chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@drop_missing(correct)
		@group_by(prolific_pid, session)
		@summarize(acc = mean(correct))
		@ungroup
		unstack([:prolific_pid], :session, :acc)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Homebase prediction accuracy",
		correct_r=false
	)
	fig
end

# ╔═╡ 1e0e87db-13bc-46db-be5d-7d2dfba7055f
let
	retest_df = @chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@mutate(half = ifelse(~denserank(trial) <= maximum(~denserank(trial))/2, "x", "y"))
		@drop_missing(correct)
		@group_by(prolific_pid, session, half)
		@summarize(acc = mean(correct))
		@ungroup
		unstack([:prolific_pid, :session], :half, :acc)
		@mutate(acc_change = y - x)
		select(Not(Cols(:x, :y)))
		unstack([:prolific_pid], :session, :acc_change)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Prediction accuracy change",
		correct_r=false
	)
	fig
end

# ╔═╡ b952e83b-85c3-4c77-b132-6bbe4f532ac4
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_confidence")
		@drop_missing(response)
		@group_by(session, trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@ungroup()
		@arrange(session, trial)
		data(_) * (mapping(:trial => nonnumeric, :lower, :upper; col=:session) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response, col=:session) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Confidence rating (0-4)"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 8dcad11f-27b8-409f-a7ab-1346a2ae3a81
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@summary(rt)
end

# ╔═╡ 58c06308-73a6-4628-9449-d808eb223ed5
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@group_by(session, trial, target)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper, color=:target, col=:session) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc, color=:target, col=:session) * visual(ScatterLines, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ 022e8a60-902a-4abc-9ee8-d7d5e6e1e3d5
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@group_by(session, reward_amount)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
end

# ╔═╡ d0ed4273-f5ed-478a-ba8b-581d9edff4a4
let
	retest_df = @chain control_task_data begin
		@drop_missing(response)
		@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
		@group_by(prolific_pid, session)
		@summarize(acc = mean(skipmissing(correct_choice)))
		@ungroup
		unstack([:prolific_pid], :session, :acc)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Correct choice rate",
		correct_r=false
	)
	fig
end

# ╔═╡ b543623b-f867-4b6d-a654-33b0256bb630
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(correct)
	@group_by(session, trial, target)
	@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper, color=:target, col=:session) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc, color=:target, col=:session) * visual(ScatterLines, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial reward rate"), figure=(;size=(800, 400)))
end

# ╔═╡ f9e75bdb-d096-42c9-8ce9-27685b1f72ab
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(correct)
	@group_by(session, reward_amount)
	@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
	@ungroup
end

# ╔═╡ 15040805-9025-4f8e-8a1d-7ea006261b83
let
	retest_df = @chain control_task_data begin
		@group_by(prolific_pid, session)
		@summarize(acc = mean(skipmissing(correct)))
		@ungroup
		unstack([:prolific_pid], :session, :acc)
		dropmissing()
	end

	fig=Figure(;size=(8, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=retest_df,
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="Test-retest: Reward rate",
		correct_r=false
	)
	fig
end

# ╔═╡ 98219665-bba0-44f3-8c5b-332da11de34d
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_controllability")
		@drop_missing(response)
		@group_by(session, trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@ungroup()
		@arrange(session, trial)
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower, col=:session) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response, col=:session) * visual(ScatterLines))
		draw(;figure=(;size=(800, 400)))
	end
end

# ╔═╡ cd5210cb-dbce-488f-93ef-f6e2e42d6118
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses, correct)
		@group_by(prolific_pid, session, correct, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, session, correct, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col = :session, row = :correct, color = :session) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col = :session, row = :correct, color = :session, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = [false => "Incorrect choice", true => "Correct choice"]), Col = (; categories = ["1" => "Session 1", "2" => "Session 2"]), Color = (; legend = false)); axis=(; xlabel = "Current strength", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ e1bd1abf-ab0b-4bbf-a9c7-76891414cd0a
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	retest_df = @chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses)
		@group_by(prolific_pid, session, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, session, current)
 		groupby([:prolific_pid, :session])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β0)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β0",
		correct_r=false
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(retest_df, [:prolific_pid], :session, :β_current)),
		xlabel="Session 1",
		ylabel="Session 2",
		xcol=Symbol(string(1)),
		ycol=Symbol(string(2)),
		subtitle="β[Current]",
		correct_r=false
	)
	Label(fig[0,:], "Presses ~ β0 + β1 * Current strength")
	fig
end

# ╔═╡ 85d9ed09-7848-4749-83c0-310e6b6fdac0
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses)
		@group_by(prolific_pid, session, reward_amount, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, session, reward_amount, current)
		data(_) * (
			mapping(:reward_amount, :trial_presses, row = :session, col = :current => nonnumeric, color = :session) * visual(RainClouds) +
			mapping(:reward_amount, :trial_presses, row = :session, col = :current => nonnumeric, color = :session, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = ["1" => "Session 1", "2" => "Session 2"]), Color = (; legend = false)); axis=(; xlabel = "Reward amount", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ 75d53fbd-5af6-4b29-a952-b5f428c98d20
let
	instr_g = @chain jspsych_data begin
		semijoin(p_no_double_take, on=:record_id)
		filter(x -> !ismissing(x.trialphase) && contains(x.trialphase, r"control_preload|control_instruction.*"), _)
		groupby([:session, :record_id])
		combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :instr_dur)
		groupby(:session)
	end
	instr = combine(x -> describe(x, cols=:instr_dur), instr_g)

	explore_g = @chain jspsych_data begin
		semijoin(p_no_double_take, on=:record_id)
		filter(x -> !ismissing(x.trialphase) && x.trialphase in ["control_explore", "control_predict_homebase", "control_controllability", "control_confidence"], _)
		groupby([:session, :record_id])
		combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :explore_dur)
		groupby(:session)
	end
	explore = combine(x -> describe(x, cols=:explore_dur), explore_g)

	reward_g = @chain jspsych_data begin
		semijoin(p_no_double_take, on=:record_id)
		filter(x -> !ismissing(x.trialphase) && x.trialphase in ["control_reward_prompt", "control_reward"], _)
		groupby([:session, :record_id])
		combine(:time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :reward_dur)
		groupby(:session)
	end
	reward = combine(x -> describe(x, cols=:reward_dur), reward_g)

	@bind_rows(instr, explore, reward)
end

# ╔═╡ 734a8fb3-dcdf-4e51-8c16-1bb0f2ee7808
spearman_brown(
r;
n = 2 # Number of splits
) = (n * r) / (1 + (n - 1) * r)

# ╔═╡ Cell order:
# ╠═656eb242-31f0-11f0-3950-5b952c0f4e01
# ╠═7b2bbc61-acc2-49d4-9195-4c60cfcb894a
# ╠═c733e42b-92dd-4615-860d-78b1efe22d41
# ╟─32ebb683-0618-49b5-9eea-acd005d8f336
# ╠═443091f5-f46b-4b7c-b7c8-b98f6d9ceba9
# ╠═5a136247-6e4b-4f6a-b76c-ba380f18d68c
# ╠═47f6b0b5-ea48-4d47-929e-388991fd52a4
# ╠═b019dcd5-8eaf-4759-83ed-cc343718fd17
# ╠═9ebce139-53fd-43de-bb82-feaf1c759910
# ╟─cae69da2-ec7f-4df6-a69d-d0677185d108
# ╟─a590ab5d-647a-4d4f-b3d2-9ef03e7b7f73
# ╟─8a0f53d3-2054-4845-b96d-29399b5aa340
# ╟─517492c4-4055-4317-bd45-e0e2fb77991e
# ╠═d77efaa6-3b94-4947-b474-97d77719d80f
# ╟─d2892577-9e9c-4c23-8b05-87c00b1962f6
# ╠═bfd770f5-3142-4a73-b39f-012d1bdb5de4
# ╟─272ba203-34cb-4ffc-9b35-4e4d23f1c59d
# ╟─5230eada-33ba-413f-88a0-96f76164bc12
# ╟─95bb623c-090d-4338-aa3f-7b31d47d87b2
# ╠═ad6c8a9d-f0a7-4b6f-b012-dbe33b894a12
# ╟─12b6b34d-cfc5-48cd-8d50-150c4e388405
# ╟─8e186be5-f2a2-4285-a62a-9c021b2dd172
# ╟─827751f8-ef30-472b-84f5-631fb212ba14
# ╠═d889a3bb-04cc-433f-94e1-a8c71739f82f
# ╟─c39a9349-ebe3-4b64-9775-a6dd04ac0847
# ╟─b54f2cb1-5dbb-4ebb-9073-78e1209cda15
# ╟─bbde0375-f1a9-47e3-b7d0-885a4e0b8548
# ╟─abe9466b-e4cc-4803-afdf-ada8f482a732
# ╟─6db5310b-7e8b-42e9-bf8a-38c862e9396c
# ╟─009447fb-3719-49df-af9d-06f2f26908f3
# ╟─5c38974b-06a9-4497-8ed8-0cf30e313abd
# ╟─75d53fbd-5af6-4b29-a952-b5f428c98d20
# ╟─cc52da42-9561-4619-8d6a-237f9ac629e8
# ╟─a395432f-0dba-4a06-bbe2-d00c596c455e
# ╟─9c4338bb-5e7b-443b-afd4-24e87349af9e
# ╟─2b29cc46-f722-4cf3-b1bf-9bb6310b2457
# ╟─a83d1fda-81fe-4305-ab8e-b6c4fbce6749
# ╟─db0ef21a-1b00-4eee-983b-f3f553730f14
# ╟─8850fa85-de8c-4328-9df0-dde761852e7a
# ╟─fa6064ce-626b-4309-9183-b129e925058b
# ╟─dc8445f6-965c-4d57-af9d-136059fe72ec
# ╟─143f8d33-fabd-4af1-b7ba-69984084a533
# ╟─fa930366-523b-467b-8c3d-5fdd77565064
# ╟─fb6d4fdf-ec68-46a2-b5dc-07f0e7cc9a40
# ╟─8f1ab7ba-0fa0-4b0f-9b69-9d8440e60f12
# ╟─d0b7207c-4e58-48ea-af3d-2503134b3f5b
# ╟─1e0e87db-13bc-46db-be5d-7d2dfba7055f
# ╟─b952e83b-85c3-4c77-b132-6bbe4f532ac4
# ╟─188a9121-a69a-4e20-8600-1a8d7ca7040b
# ╟─8dcad11f-27b8-409f-a7ab-1346a2ae3a81
# ╟─58c06308-73a6-4628-9449-d808eb223ed5
# ╟─022e8a60-902a-4abc-9ee8-d7d5e6e1e3d5
# ╟─d0ed4273-f5ed-478a-ba8b-581d9edff4a4
# ╟─b543623b-f867-4b6d-a654-33b0256bb630
# ╟─f9e75bdb-d096-42c9-8ce9-27685b1f72ab
# ╟─15040805-9025-4f8e-8a1d-7ea006261b83
# ╟─98219665-bba0-44f3-8c5b-332da11de34d
# ╟─cd5210cb-dbce-488f-93ef-f6e2e42d6118
# ╟─e1bd1abf-ab0b-4bbf-a9c7-76891414cd0a
# ╠═85d9ed09-7848-4749-83c0-310e6b6fdac0
# ╟─f74eea14-748b-4b3c-b887-bc9485ecfea3
# ╟─8312e274-4cf4-4ad2-906d-e357ef5d4755
# ╟─8f2909c4-7746-4994-b97e-6acc87aa84a1
# ╟─734a8fb3-dcdf-4e51-8c16-1bb0f2ee7808
