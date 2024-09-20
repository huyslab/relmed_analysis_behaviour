### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ da2aa306-75f9-11ef-2592-2be549c73d82
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP
	include("fetch_preprocess_data.jl")
	nothing
end

# ╔═╡ 6eba46dc-855c-47ca-8fa9-8405b9566809
jspsych_data = let
	
	jspsych_json, records = get_REDCap_data("pilot2"; file_field = "file_data")
	
	jspsych_data = REDCap_data_to_df(jspsych_json, records)
end

# ╔═╡ f47e6aba-00ea-460d-8310-5b24ed7fe336
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
	debrief = filter(x -> !ismissing(x.trialphase) && x.trialphase in ["debrief_text", "debrief_likert"], data)

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
	likert_expanded = 
		DataFrame([JSON.parse(row.debrief_likert) for row in eachrow(debrief)])

	text_expanded = 
		DataFrame([JSON.parse(row.debrief_text) for row in eachrow(debrief)])

	# hcat together
	return hcat(debrief[!, Not([:debrief_likert, :debrief_text])], likert_expanded, text_expanded)
end

# ╔═╡ d203faab-d4ea-41b2-985b-33eb8397eecc
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
		[:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		:total_presses => (x -> length(filter(y -> !ismissing(y), x))) => :n_trial_vigour,
		:block => (x -> filter(y -> typeof(y) == Int64, x) |> unique |> length) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ e35effd6-5c62-48aa-8932-872c7af50d7b
summarize_participation(jspsych_data)

# ╔═╡ Cell order:
# ╠═da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═6eba46dc-855c-47ca-8fa9-8405b9566809
# ╠═e35effd6-5c62-48aa-8932-872c7af50d7b
# ╠═d203faab-d4ea-41b2-985b-33eb8397eecc
# ╠═f47e6aba-00ea-460d-8310-5b24ed7fe336
