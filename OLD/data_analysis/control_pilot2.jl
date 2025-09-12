### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ 237a05f6-9e0e-11ef-2433-3bdaa51dbed4
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
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("osf_utils.jl")
	include("vigour_utils.jl")
	nothing
end

# ╔═╡ 3251261f-38da-451c-abfa-48df4c40601d
TableOfContents(title="📚 Table of Contents", indent=true, depth=4, aside=true)

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

	spearman_brown(
	r;
	n = 2 # Number of splits
	) = (n * r) / (1 + (n - 1) * r)
end

# ╔═╡ d5811081-d5e2-4a6e-9fc9-9d70332cb338
md"""## Participant management"""

# ╔═╡ f9cbc7fa-5998-4500-a5b7-9f93258e7608
begin
	raw_control_task_data, raw_control_report_data, jspsych_data = load_control_pilot2_data(; force_download = false, session = "1")
	nothing
end

# ╔═╡ bc1db6ac-0ed2-4d04-a711-823627df6173
begin
	p_careless = @chain jspsych_data begin
		@group_by(prolific_pid)
		@summarize(n_quiz_fail = sum(skipmissing(trialphase == "control_instruction_quiz_failure")))
		@mutate(careless = n_quiz_fail >= ~median(n_quiz_fail))
	end
end

# ╔═╡ 821e95eb-79af-4168-83a4-3a2a2e791939
begin
	info_files = filter(x -> endswith(x, ".csv"), readdir(joinpath("data", "prolific_participant_info"); join = true))
	info_data = @chain info_files begin
		CSV.read(DataFrame, select = ["Participant id", "Age"], types=Dict(:Age=>String))
		@rename(prolific_pid = var"Participant id", age = var"Age")
		@filter(age != "CONSENT_REVOKED")
		@mutate(age = as_float(age))
	end
end

# ╔═╡ ddd696e7-a014-41b5-a5c3-2b7b2d38eb2b
unique(jspsych_data.version)

# ╔═╡ a9271e63-6457-47c0-99c4-07304bb31a93
md"""
## Control task
"""

# ╔═╡ e1ff3af0-4e8b-4bf7-9c30-cf227853d7d3
# ╠═╡ disabled = true
#=╠═╡
begin
	@save "control_task_data.jld2" {compress=true} control_task_data
	@save "control_report_data.jld2" {compress=true} control_report_data
end
  ╠═╡ =#

# ╔═╡ af4bb053-7412-4b00-bb3c-5f1eb8cd9e5b
# ╠═╡ disabled = true
#=╠═╡
begin
	CSV.write("control_task_data.csv", control_task_data; transform=(col, val) -> something(val, missing))
	CSV.write("control_report_data.csv", control_report_data; transform=(col, val) -> something(val, missing))
end
  ╠═╡ =#

# ╔═╡ b2d724e5-146a-4812-8431-a77893ea4735
begin
	threshold_df = (; y = [6, 12, 18], threshold = ["Low", "Mid", "High"])
	spec2 = data(threshold_df) * mapping(:y, color = :threshold) * visual(HLines)
end

# ╔═╡ b4c02bfd-3252-441f-a2b8-6178beb2b144
md"""
## Helper functions
"""

# ╔═╡ 83c4cd4a-616a-4762-8dff-f6439fd948f7
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

# ╔═╡ 5c1c7680-d743-4488-a8fc-c81cb23cb87e
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
		:n_control_trials => (x -> maximum(skipmissing(x); init = 0)) => :n_trial_control,
		[:trialphase, :correct] => ((p, c) -> sum(c[.!ismissing.(p) .&& p .== "control_reward_feedback"]) * 5 / 100) => :control_bonus,
		:n_warnings => maximum => :n_warnings,
		:time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration,
		:trialphase => (x -> sum(skipmissing(x .== "control_instruction_quiz_failure"), init=0)) => :n_quiz_failure
	)

	# Compute totla bonus
	insertcols!(participants, 
		:total_bonus => ifelse.(
			ismissing.(participants.control_bonus),
			fill(0., nrow(participants)),
			participants.control_bonus
		)
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ 8c7c8ee7-86e6-48d4-9a8c-e24b4c35e239
begin
	p_sum = summarize_participation(jspsych_data)
	@info "# Valid data samples: $(sum(skipmissing(p_sum.finished)))"
	@info "# Participants who finished control: $(sum(p_sum.n_trial_control .== 144))"
end

# ╔═╡ 06519325-9455-45e3-a566-b76ede99808d
begin
	finished_participants = filter(x -> x.n_trial_control .== 144, p_sum)
	control_task_data = semijoin(raw_control_task_data, finished_participants, on=:prolific_pid)
	control_report_data = semijoin(raw_control_report_data, finished_participants, on=:prolific_pid)
end

# ╔═╡ 021062cb-b9f5-46bd-addb-de68d122531e
@assert all(combine(groupby(control_task_data, :prolific_pid), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created incorrectly in chronological order"

# ╔═╡ e3392191-71cd-4a5d-b2a4-5e2a72b111d9
control_task_data

# ╔═╡ ef837154-28e6-4e50-bec4-efe04f45a6cd
combine(groupby(control_task_data, :prolific_pid), :time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :duration_m) |>
describe

# ╔═╡ 3fc647cb-4e05-490a-b1b4-3240db3a1823
begin
	@chain control_task_data begin
		@filter(trialphase == "control_explore")
		@mutate(trial_stage = ifelse(trial <= maximum(trial)/2, "Trial 1-36", "Trial 37-72"))
		@drop_missing(trial_presses)
		@group_by(prolific_pid, trial_stage, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, trial_stage, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col=:trial_stage, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(spec2 + _; axis=(; xlabel = "Current strength", ylabel = "Presses (in explore trials)"))
	end
end

# ╔═╡ dcf49989-910b-4744-942a-66124e694944
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	split_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_number = ~denserank(trial))
		@mutate(half = ifelse(trial_number <= maximum(trial_number)/2, "x", "y"))
		groupby([:prolific_pid, :half])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(split_df, [:prolific_pid], :half, :β0)),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle="β0",
		correct_r=true
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(split_df, [:prolific_pid], :half, :β_current)),
		xlabel="Trial 1-36",
		ylabel="Trial 37-72",
		xcol=:x,
		ycol=:y,
		subtitle="β[Current]",
		correct_r=true
	)
	fig
end

# ╔═╡ 7b603b2c-383c-4ca7-bb69-78aa57ffc6ec
let
	glm_coef(data) = coef(lm(@formula(trial_presses ~ current), data))
	
	split_df = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		@drop_missing(trial_presses)
		@mutate(trial_number = ~denserank(trial))
		@mutate(half = if_else(trial_number % 2 === 0, "x", "y"))
		groupby([:prolific_pid, :half])
		combine(AsTable([:trial_presses, :current]) => (x -> [glm_coef(x)]) => [:β0, :β_current])
	end

	fig=Figure(;size=(12, 6) .* 144 ./ 2.54)
	workshop_reliability_scatter!(
		fig[1,1];
		df=dropmissing(unstack(split_df, [:prolific_pid], :half, :β0)),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:x,
		ycol=:y,
		subtitle="β0",
		correct_r=true
	)
	workshop_reliability_scatter!(
		fig[1,2];
		df=dropmissing(unstack(split_df, [:prolific_pid], :half, :β_current)),
		xlabel="Even trials",
		ylabel="Odd trials",
		xcol=:x,
		ycol=:y,
		subtitle="β[Current]",
		correct_r=true
	)
	fig
end

# ╔═╡ c8975d53-fcec-41ee-9325-098feda9833d
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@mutate(prolific_pid = str_trunc(prolific_pid, 9))
		# @drop_missing(correct)
		data(_) * mapping(:trial => nonnumeric, :correct, layout=:prolific_pid) * visual(ScatterLines)
		draw(;axis=(;xlabel = "Trial", ylabel = "Home base predict acc."), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 0a5622ee-2668-498f-8275-20cfda686e43
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@drop_missing(correct)
		@left_join(p_careless)
		@group_by(careless, trial, ship)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		@ungroup
		data(_) * (mapping(:trial, :lower, :upper, color=:ship, col=:careless) * visual(Band, alpha = 0.1) + mapping(:trial, :acc, color=:ship, col=:careless) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Home base predict acc."), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 32577c4d-6976-4854-827b-97b57f25ba80
begin
	explore_history = @chain control_task_data begin
		@filter(trialphase == "control_explore")
		transform(:trial => (x -> convert.(Int, floor.((denserank(x .+ 6) .- 1) / 12)) .+ 1) => :block)
		@group_by(prolific_pid, block, ship_color)
		@summarize(n = n(), n_control = sum(control_rule_used == "control"))
		@ungroup()
		@drop_missing(ship_color)
		@group_by(prolific_pid, ship_color)
		@mutate(n_cumu = cumsum(n), n_control_cumu = cumsum(n_control))
		@ungroup()
	end

	predict_history = @chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		transform(:trial => (x -> convert.(Int, floor.((denserank(x) .- 1) / 4)) .+ 1) => :block)
		@select(prolific_pid, block, ship_color = ship, correct)
	end

	history = leftjoin(explore_history, predict_history, on = [:prolific_pid, :block, :ship_color])

	@chain history begin
		@group_by(block, ship_color)
		@summarize(n_avg = mean(n_cumu), n_control_avg = mean(n_control_cumu), acc = mean(correct))
		@ungroup()
		@drop_missing()
		data(_) * mapping(:n_control_avg, :acc, color=:ship_color) * (visual(Scatter) + linear())
		draw()
	end
end

# ╔═╡ bc08a966-4236-4a5e-9539-5913f8f64651
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@mutate(trial_stage = ifelse(trial <= maximum(trial)/2, "Trial 1-72", "Trial 73-144"))
		@drop_missing(correct)
		@group_by(prolific_pid, trial_stage)
		@summarize(acc = mean(correct))
		@ungroup()
		data(_) * mapping(:acc, col=:trial_stage) * visual(Hist)
		draw()
		end
end

# ╔═╡ bb771903-c617-42c2-b3ae-7e48b09aacf9
@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@mutate(trial_stage = ifelse(trial <= maximum(trial)/2, "Trial 1-72", "Trial 73-144"))
		@drop_missing(correct)
		@group_by(prolific_pid, trial_stage)
		@summarize(acc = mean(correct))
		@ungroup()
		@pivot_wider(names_from = "trial_stage", values_from = "acc")
end

# ╔═╡ e16cfc88-b1f3-4be4-b248-8dde073d9c8c
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_confidence")
		@drop_missing(response)
		@arrange(trial)
		data(_) * (mapping(:trial => nonnumeric, :response, color=:prolific_pid) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Confidence rating (0-4)"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ bf50e3bd-c697-4e8b-93ff-558ec99711b0
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_confidence")
		@drop_missing(response)
		@group_by(trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@arrange(trial)
		data(_) * (mapping(:trial => nonnumeric, :lower, :upper) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response) * visual(ScatterLines))
		draw(;axis=(;xlabel = "Trial", ylabel = "Confidence rating (0-4)"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 85fd37d4-39f5-4c31-92a9-2c597a1c790a
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(correct)
		@group_by(trial, island_viable, current)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		@ungroup
		data(_) * (mapping(:trial => nonnumeric, :lower, :upper) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc) * visual(Lines, linestyle=:dot) + mapping(:trial => nonnumeric, :acc, color=:current => nonnumeric, marker=:island_viable) * visual(Scatter, markersize = 12))
		draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial success rate"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ c75dfcb1-4aa9-47a1-952a-6553b4035a7b
let
	ship_current_map = (;yellow = "orange", red = "grape", blue = "coconut", green = "banana")
	df = @chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(correct, ship_color)
	end
	transform!(df, [:left_viable, :right, :left] => ((x, y, z) -> ifelse.(x, y, z)) => :wrong_option)
	transform!(df, [:wrong_option, :near] => ByRow((x, y) -> ship_current_map[Symbol(x)] == y) => :wrong_match_homebase)
	combine(groupby(df, :wrong_match_homebase), :correct => (x -> (1 - mean(x))) => :prop_incorrect)
end

# ╔═╡ dff932c8-1317-4566-88c6-7b33cbc04b3f
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@drop_missing(correct_choice)
	@group_by(trial, island_viable, current)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc) * visual(Lines, linestyle=:dot) + mapping(:trial => nonnumeric, :acc, color=:current => nonnumeric, marker=:island_viable) * visual(Scatter, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ 48475e7e-60c5-4847-b6f2-4e6027e7dc10
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@drop_missing(correct_choice)
	@group_by(trial, target)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * (mapping(:trial => nonnumeric, :lower, :upper, color=:target) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc, color=:target) * visual(ScatterLines, markersize = 12))
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ 98d36689-8eeb-41c4-a321-0ab039882212
@chain control_task_data begin
	@filter(trialphase == "control_reward")
	@drop_missing(response)
	@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
	@drop_missing(correct_choice)
	@group_by(trial, target)
	@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
	@ungroup
	data(_) * mapping(:trial, :acc, color=:target) * (visual(Scatter, markersize = 12) + linear())
	draw(;axis=(;xlabel = "Trial", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
end

# ╔═╡ 810612c9-9b20-4076-b3b0-9406df690862
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(correct)
		@group_by(island_viable, current)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		@ungroup
		data(_) * (mapping(:current => nonnumeric, :acc, dodge = :island_viable, color = :island_viable) * visual(BarPlot))
		draw(;axis=(;xlabel = "Current", ylabel = "Reward trial success rate"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ 1e0dc9cc-5022-44b0-b52e-a5b7f5521044
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(response)
		@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
		@drop_missing(correct_choice)
		@group_by(island_viable, current)
		@summarize(acc = mean(correct_choice), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)))
		@ungroup
		data(_) * (mapping(:current => nonnumeric, :acc, dodge = :island_viable, color = :island_viable) * visual(BarPlot))
		draw(;axis=(;xlabel = "Current", ylabel = "Reward trial correct choice rate"), figure=(;size=(800, 400)))
	end
end

# ╔═╡ bbd5ba75-263b-481f-8be7-e6b575139a1e
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(response)
		@mutate(correct_choice = ifelse(response == "left", left_viable, right_viable))
		@drop_missing(correct_choice)
		@group_by(island_viable, current)
		@summarize(acc = mean(correct_choice), lower = mean(correct_choice) - std(correct_choice)/sqrt(length(correct_choice)), upper = mean(correct_choice) + std(correct_choice)/sqrt(length(correct_choice)))
		@ungroup
		@arrange(island_viable, current)
	end
end

# ╔═╡ df2a8e6d-5c18-41c3-a2dd-b17f09b49428
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses, correct)
		@group_by(prolific_pid, correct, island_viable, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, correct, island_viable, current)
		data(_) * (
			mapping(:current => nonnumeric, :trial_presses, col = :island_viable, row = :correct, color = :island_viable) * visual(RainClouds) +
			mapping(:current => nonnumeric, :trial_presses, col = :island_viable, row = :correct, color = :island_viable, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = [false => "Incorrect choice", true => "Correct choice"]), Col = (; categories = [false => "Island unviable", true => "Island viable"]), Color = (; legend = false)); axis=(; xlabel = "Current strength", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ 074ca439-f95d-43bb-a791-6d21a8ad34bf
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(trial_presses, correct)
		@group_by(prolific_pid, correct, island_viable, current)
		@summarize(trial_presses = mean(trial_presses))
		@ungroup()
		@arrange(prolific_pid, correct, island_viable, current)
		data(_) * (
			mapping(:island_viable, :trial_presses, col = :current => nonnumeric, row = :correct, color = :current => nonnumeric) * visual(RainClouds) +
			mapping(:island_viable, :trial_presses, col = :current => nonnumeric, row = :correct, color = :current => nonnumeric, group=:prolific_pid) * visual(Lines, alpha = 0.1)
		)
		draw(_, scales(Row = (; categories = [false => "Incorrect choice", true => "Correct choice"]), Color = (; legend = false)); axis=(; xlabel = "Island viable?", ylabel = "Presses (in reward trials)"), figure=(;size=(600, 600)))
	end
end

# ╔═╡ bd4b266a-c4aa-4851-a601-e41df751059c
begin
	transform!(control_report_data, :response => (x -> ifelse.(x == nothing, missing, x)) => :response)
	@chain control_report_data begin
		@filter(trialphase == "control_controllability")
		@drop_missing(response)
		@group_by(trial)
		@summarize(response = mean(response), upper = mean(response) + std(response)/sqrt(length(response)), lower = mean(response) - std(response)/sqrt(length(response)))
		@arrange(trial)
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response) * visual(ScatterLines))
		draw(;figure=(;size=(800, 400)))
	end
end

# ╔═╡ 2d25ef0c-abe2-485b-9629-a0ee3eb481ba
p_sum

# ╔═╡ 765d05c0-0679-4f26-b201-af2aa0bf3fa3
describe(p_sum)

# ╔═╡ 9d14e29c-8136-4294-ba50-53b1cb480bc0
begin
	@chain p_sum begin
		@filter(n_quiz_failure < 20)
		@pivot_longer(control_enjoy:control_clear, names_to = "accept_var", values_to = "accept_val")
		stack([:n_quiz_failure, :n_warnings], variable_name = :sanity_var, value_name = :sanity_val)
		dropmissing([:accept_val, :sanity_val])
		data(_) * mapping(:sanity_val, :accept_val, col = :accept_var, row = :sanity_var) * (visual(Scatter) + linear())
		draw(facet = (; linkxaxes = :none, linkyaxes = :all))
	end
end

# ╔═╡ 8c3efe36-5fdb-4524-907c-7770a2dfb36c
begin
	@chain p_sum begin
		@left_join(info_data)
		@filter(age < 75)
		stack([:n_quiz_failure, :n_warnings, :control_enjoy, :control_difficulty, :control_clear], variable_name=:rating_var, value_name=:rating_val)
		dropmissing([:age, :rating_val])
		data(_) * mapping(:age, :rating_val, layout=:rating_var) * (visual(Scatter) + linear())
		draw(facet = (; linkxaxes = :all, linkyaxes = :none))
	end
end

# ╔═╡ 47dec249-b853-4860-afcb-d8f4b0cbac81
foreach(row -> print(row.prolific_pid * "\r\n"), eachrow(p_sum[p_sum.n_trial_control .== 144, [:prolific_pid]]))

# ╔═╡ 8013dd07-e36b-4449-addf-b5fdbeed3f75
foreach(row -> print(row.prolific_pid * "," * as_string(row.total_bonus) * "\r\n"), eachrow(p_sum[:, [:prolific_pid, :total_bonus]]))

# ╔═╡ fc6afa82-b9fe-41f5-a2ca-c5cb38d53b73
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

# ╔═╡ Cell order:
# ╠═237a05f6-9e0e-11ef-2433-3bdaa51dbed4
# ╠═3251261f-38da-451c-abfa-48df4c40601d
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═f9cbc7fa-5998-4500-a5b7-9f93258e7608
# ╠═bc1db6ac-0ed2-4d04-a711-823627df6173
# ╠═821e95eb-79af-4168-83a4-3a2a2e791939
# ╠═8c7c8ee7-86e6-48d4-9a8c-e24b4c35e239
# ╠═06519325-9455-45e3-a566-b76ede99808d
# ╠═ddd696e7-a014-41b5-a5c3-2b7b2d38eb2b
# ╠═2d25ef0c-abe2-485b-9629-a0ee3eb481ba
# ╠═765d05c0-0679-4f26-b201-af2aa0bf3fa3
# ╠═9d14e29c-8136-4294-ba50-53b1cb480bc0
# ╠═8c3efe36-5fdb-4524-907c-7770a2dfb36c
# ╠═47dec249-b853-4860-afcb-d8f4b0cbac81
# ╠═8013dd07-e36b-4449-addf-b5fdbeed3f75
# ╟─a9271e63-6457-47c0-99c4-07304bb31a93
# ╠═021062cb-b9f5-46bd-addb-de68d122531e
# ╠═e3392191-71cd-4a5d-b2a4-5e2a72b111d9
# ╠═ef837154-28e6-4e50-bec4-efe04f45a6cd
# ╠═e1ff3af0-4e8b-4bf7-9c30-cf227853d7d3
# ╠═af4bb053-7412-4b00-bb3c-5f1eb8cd9e5b
# ╠═b2d724e5-146a-4812-8431-a77893ea4735
# ╠═3fc647cb-4e05-490a-b1b4-3240db3a1823
# ╠═dcf49989-910b-4744-942a-66124e694944
# ╠═7b603b2c-383c-4ca7-bb69-78aa57ffc6ec
# ╠═c8975d53-fcec-41ee-9325-098feda9833d
# ╠═0a5622ee-2668-498f-8275-20cfda686e43
# ╠═32577c4d-6976-4854-827b-97b57f25ba80
# ╠═bc08a966-4236-4a5e-9539-5913f8f64651
# ╠═bb771903-c617-42c2-b3ae-7e48b09aacf9
# ╠═e16cfc88-b1f3-4be4-b248-8dde073d9c8c
# ╠═bf50e3bd-c697-4e8b-93ff-558ec99711b0
# ╠═85fd37d4-39f5-4c31-92a9-2c597a1c790a
# ╠═c75dfcb1-4aa9-47a1-952a-6553b4035a7b
# ╠═dff932c8-1317-4566-88c6-7b33cbc04b3f
# ╠═48475e7e-60c5-4847-b6f2-4e6027e7dc10
# ╠═98d36689-8eeb-41c4-a321-0ab039882212
# ╠═810612c9-9b20-4076-b3b0-9406df690862
# ╠═1e0dc9cc-5022-44b0-b52e-a5b7f5521044
# ╠═bbd5ba75-263b-481f-8be7-e6b575139a1e
# ╠═df2a8e6d-5c18-41c3-a2dd-b17f09b49428
# ╠═074ca439-f95d-43bb-a791-6d21a8ad34bf
# ╠═bd4b266a-c4aa-4851-a601-e41df751059c
# ╟─b4c02bfd-3252-441f-a2b8-6178beb2b144
# ╠═5c1c7680-d743-4488-a8fc-c81cb23cb87e
# ╠═83c4cd4a-616a-4762-8dff-f6439fd948f7
# ╠═fc6afa82-b9fe-41f5-a2ca-c5cb38d53b73
