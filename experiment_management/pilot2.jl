### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═╡ show_logs = false
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Dates, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 6eba46dc-855c-47ca-8fa9-8405b9566809
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
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

	participants.total_bonus = participants.vigour_bonus .+ participants.PILT_bonus

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ 4eb3aaed-6028-49a2-9f13-4e915ee2701c
p_sum = summarize_participation(jspsych_data)

# ╔═╡ 99deaa70-9bec-4da4-8f0d-1d28ade2f191
filter(x -> !ismissing(x.finished) && x.finished, p_sum)

# ╔═╡ fd03411b-d9ba-4c75-ad74-53fa046f0d49
mean(skipmissing(p_sum.vigour_bonus))

# ╔═╡ Cell order:
# ╠═da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═6eba46dc-855c-47ca-8fa9-8405b9566809
# ╠═4eb3aaed-6028-49a2-9f13-4e915ee2701c
# ╠═99deaa70-9bec-4da4-8f0d-1d28ade2f191
# ╠═fd03411b-d9ba-4c75-ad74-53fa046f0d49
# ╠═d203faab-d4ea-41b2-985b-33eb8397eecc
# ╠═f47e6aba-00ea-460d-8310-5b24ed7fe336
