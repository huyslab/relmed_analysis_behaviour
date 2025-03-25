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

# ╔═╡ f9cbc7fa-5998-4500-a5b7-9f93258e7608
begin
	PILT_data, WM_data, LTM_data, WM_test_data, LTM_test_data, max_press_data, control_task_data, control_report_data, jspsych_data = load_pilot8_data(; force_download = false, return_version = "0.2")
	nothing
end

# ╔═╡ a9271e63-6457-47c0-99c4-07304bb31a93
md"""
## Control task
"""

# ╔═╡ 021062cb-b9f5-46bd-addb-de68d122531e
@assert all(combine(groupby(control_task_data, :prolific_pid), [:time_elapsed, :trial] => ((x, y) -> all(denserank(x) .== denserank(y))) => :in_order)[:, "in_order"]) "Trial numbers are created incorrectly in chronological order"

# ╔═╡ ef837154-28e6-4e50-bec4-efe04f45a6cd
combine(groupby(control_task_data, :prolific_pid), :time_elapsed => (x -> (maximum(x) - minimum(x))/60000) => :duration_m) |>
describe

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

# ╔═╡ 0a5622ee-2668-498f-8275-20cfda686e43
begin
	@chain control_task_data begin
		@filter(trialphase == "control_predict_homebase")
		@drop_missing(correct)
		@group_by(trial)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc) * visual(ScatterLines))
		draw
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
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :response) * visual(ScatterLines))
		draw
	end
end

# ╔═╡ 85fd37d4-39f5-4c31-92a9-2c597a1c790a
begin
	@chain control_task_data begin
		@filter(trialphase == "control_reward")
		@drop_missing(correct)
		@group_by(trial)
		@summarize(acc = mean(correct), upper = mean(correct) + std(correct)/sqrt(length(correct)), lower = mean(correct) - std(correct)/sqrt(length(correct)))
		data(_) * (mapping(:trial => nonnumeric, :upper, :lower) * visual(Band, alpha = 0.1) + mapping(:trial => nonnumeric, :acc) * visual(ScatterLines))
		draw
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
		draw
	end
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
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		[:trial_type, :trialphase, :block, :n_stimuli] => 
			((t, p, b, n) -> sum((t .== "PILT") .& (.!ismissing.(p) .&& p .!= "PILT_test") .& (typeof.(b) .== Int64) .& (n .== 2))) => :n_trial_PILT,
		[:trial_type, :block, :trialphase] => 
			((t, b, p) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (p .== "wm"))) => :n_trial_WM,
		[:trial_type, :block, :trialphase] => 
			((t, b, p) -> sum((t .== "PILT") .& (typeof.(b) .== Int64) .& (p .== "ltm"))) => :n_trial_LTM,
		[:block, :trial_type, :trialphase, :n_stimuli] => 
			((x, t, p, n) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (n .== 2) .& (.!ismissing.(p) .&& p .!= "PILT_test")])))) => :n_blocks_PILT,
		[:block, :trial_type, :trialphase] => 
			((x, t, p) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (p .== "wm")])))) => :n_blocks_WM,
		[:block, :trial_type, :trialphase] => 
			((x, t, p) -> length(unique(filter(y -> isa(y, Int64), x[(t .== "PILT") .& (p .== "ltm")])))) => :n_blocks_LTM,
		:trialphase => (x -> sum(skipmissing(x .∈ Ref(["control_explore", "control_predict_homebase", "control_reward"])))) => :n_trial_control,
		[:trialphase, :correct] =>
			((p, c) -> sum(c[.!ismissing.(p) .&& p .== "control_reward_feedback"]) * 5 / 100) => :control_bonus,
		:trialPresses => (x -> mean(filter(y -> !ismissing(y), x))) =>  :max_trial_presses,
		:n_warnings => maximum => :n_warnings,
		:time_elapsed => (x -> maximum(x) / 1000 / 60) => :duration
	)

	# Compute totla bonus
	insertcols!(participants, :n_trial_PILT, 
		:total_bonus => ifelse.(
			ismissing.(participants.control_bonus),
			fill(0., nrow(participants)),
			participants.control_bonus
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

# ╔═╡ 8c7c8ee7-86e6-48d4-9a8c-e24b4c35e239
begin
	p_sum = summarize_participation(jspsych_data)
	@info "# Valid data samples: $(sum(skipmissing(p_sum.finished)))"
	@info "# Participants who finished control: $(sum(p_sum.n_trial_control .== 192))"
end

# ╔═╡ 765d05c0-0679-4f26-b201-af2aa0bf3fa3
describe(p_sum)

# ╔═╡ 8013dd07-e36b-4449-addf-b5fdbeed3f75
foreach(row -> print(row.prolific_pid * "," * as_string(row.total_bonus) * "\r\n"), eachrow(p_sum[p_sum.n_trial_control .== 192, [:prolific_pid, :total_bonus]]))

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
# ╠═0d120e19-28c2-4a98-b873-366615a5f784
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═f9cbc7fa-5998-4500-a5b7-9f93258e7608
# ╠═8c7c8ee7-86e6-48d4-9a8c-e24b4c35e239
# ╠═765d05c0-0679-4f26-b201-af2aa0bf3fa3
# ╠═8013dd07-e36b-4449-addf-b5fdbeed3f75
# ╟─a9271e63-6457-47c0-99c4-07304bb31a93
# ╠═021062cb-b9f5-46bd-addb-de68d122531e
# ╠═ef837154-28e6-4e50-bec4-efe04f45a6cd
# ╠═e1ff3af0-4e8b-4bf7-9c30-cf227853d7d3
# ╠═af4bb053-7412-4b00-bb3c-5f1eb8cd9e5b
# ╠═0a5622ee-2668-498f-8275-20cfda686e43
# ╠═bf50e3bd-c697-4e8b-93ff-558ec99711b0
# ╠═85fd37d4-39f5-4c31-92a9-2c597a1c790a
# ╠═bd4b266a-c4aa-4851-a601-e41df751059c
# ╟─b4c02bfd-3252-441f-a2b8-6178beb2b144
# ╠═5c1c7680-d743-4488-a8fc-c81cb23cb87e
# ╠═83c4cd4a-616a-4762-8dff-f6439fd948f7
# ╠═fc6afa82-b9fe-41f5-a2ca-c5cb38d53b73
