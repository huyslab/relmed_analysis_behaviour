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

# ╔═╡ d5811081-d5e2-4a6e-9fc9-9d70332cb338
md"""## Participant management"""

# ╔═╡ 36b348cc-a3bf-41e7-aac9-1f6d858304a2
begin
	# Load data
	PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data,
		reversal_data, jspsych_data = load_pilot6_data()
	
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
# ╟─d5811081-d5e2-4a6e-9fc9-9d70332cb338
# ╠═36b348cc-a3bf-41e7-aac9-1f6d858304a2
# ╠═c6d0d8c2-2c26-4e9c-8c1b-a9b23d985971
# ╠═6ca0676f-b107-4cc7-b0d2-32cc345dab0d
# ╠═dc957d66-1219-4a97-be46-c6c5c189c8ba
# ╠═91f6a95c-4f2e-4213-8be5-3ca57861ed15
