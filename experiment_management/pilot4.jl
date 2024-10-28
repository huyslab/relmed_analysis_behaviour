### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ baba7ea8-9069-11ef-2bba-89fb74ddc46b
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

# ╔═╡ 57ca3929-faa6-4a95-9e4d-6c1add13b121
PLT_data, test_data, vigour_data, reversal_data, jspsych_data = load_pilot4_data()

# ╔═╡ 5d616c03-85db-4c54-baba-92d288479113
p_sum = summarize_participation(jspsych_data)

# ╔═╡ 2b4c69cb-a277-4c01-bb2e-10ce494118d7
for r in eachrow(p_sum)
	println("$(r.prolific_pid), $(round(r.total_bonus, digits = 2))")
	println(sum(filter(x -> !ismissing(x.total_bonus), p_sum).total_bonus))
end

# ╔═╡ e6ce2ef5-0bcb-45b7-a40b-0ebceff1a4c2
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
		:PIT_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :PIT_bonus,
		[:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		:total_presses => (x -> length(filter(y -> !ismissing(y), x))) => :n_trial_vigour,
		[:block, :trial_type] => ((x, t) -> length(unique(filter(y -> isa(y, Int64), x[t .== "PLT"])))) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	participants.total_bonus = participants.vigour_bonus .+ participants.PILT_bonus 
		.+ participants.PIT_bonus

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ e3ed2dd8-db2a-4725-9f78-ad338f8e0cfc
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
		occursin(r"(acceptability|debrief)(?!.*pre)", x.trialphase), data)


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
	debrief_colnames = ["acceptability_pilt", "acceptability_vigour", "debrief_vigour", "acceptability_reversal", "debrief_reversal", "debrief_instructions"]
	expanded = [
		DataFrame([JSON.parse(row[col]) for row in eachrow(debrief)]) 
			for col in debrief_colnames
		]

	expanded = hcat(expanded...)

	# hcat together
	return hcat(debrief[!, Not(debrief_colnames)], expanded)
end

# ╔═╡ Cell order:
# ╠═baba7ea8-9069-11ef-2bba-89fb74ddc46b
# ╠═57ca3929-faa6-4a95-9e4d-6c1add13b121
# ╠═5d616c03-85db-4c54-baba-92d288479113
# ╠═2b4c69cb-a277-4c01-bb2e-10ce494118d7
# ╠═e6ce2ef5-0bcb-45b7-a40b-0ebceff1a4c2
# ╠═e3ed2dd8-db2a-4725-9f78-ad338f8e0cfc
