### A Pluto.jl notebook ###
# v0.20.3

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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, Tidier, Dates
	include("fetch_preprocess_data.jl")
	nothing
end

# ╔═╡ 6eba46dc-855c-47ca-8fa9-8405b9566809
jspsych_data = let
	jspsych_json, records = get_REDCap_data("pilot3.1"; file_field = "file_data")
	
	jspsych_data = REDCap_data_to_df(jspsych_json, records)

	remove_testing!(jspsych_data)
end

# ╔═╡ bed41c93-2f7e-4f2c-bec4-d5f5cd475e0f
names(jspsych_data)

# ╔═╡ e35effd6-5c62-48aa-8932-872c7af50d7b
summarize_participation(jspsych_data)

# ╔═╡ 99ca84c3-3df1-448d-b5b6-593328611aa1
filter(x -> !ismissing(x.finished) && x.finished, summarize_participation(jspsych_data))

# ╔═╡ f221ad44-920c-4b29-a48a-ba6aa529ab45
begin
	sub_list = CSV.read("/home/jovyan/data/prolific_participant_info/prolific_export_670aadfac3deeac9d40a5788.csv", DataFrame)
	finished_df = filter(x -> !ismissing(x.finished) && x.finished && !ismissing(x.vigour_bonus), summarize_participation(jspsych_data)) |>
	x -> select(x, :prolific_pid, :vigour_bonus) |>
	x -> subset(x, :vigour_bonus => ByRow(x -> x > 0))
	finished_df = semijoin(finished_df, sub_list, on = [:prolific_pid => Symbol("Participant id")])
	for row in eachrow(finished_df)
		println(join(row, ","))
	end
end

# ╔═╡ 30aebcee-9138-46cf-8dd9-d03f93404517
sum(finished_df.vigour_bonus) * (1 + 1/3 * 1.2)

# ╔═╡ 8815162b-6592-44aa-8eb4-8c3d2e2cda59
mean(skipmissing(summarize_participation(jspsych_data).vigour_bonus))

# ╔═╡ 3de841b9-ef50-4cfa-ab5b-dd472ca01d4b
quantile(skipmissing(summarize_participation(jspsych_data).vigour_bonus), [0, 0.25, 0.5, 0.75, 1])

# ╔═╡ 431726e1-c06b-40b6-9183-107cc138a5d4
md"""
Check if there are timing issues
"""

# ╔═╡ 6ef2d203-d35e-4480-895f-cf1e553b469a
begin
	@chain jspsych_data begin
		@select(prolific_pid, version, time_elapsed, trialphase)
		@filter(!(occursin(prolific_pid, "simulate")))
		@filter(trialphase == "vigour_trial")
		@group_by(prolific_pid, version)
		@mutate(dur = [missing, ~ diff(time_elapsed)...])
		@summarize(dur_min = minimum(skipmissing(dur)),
					dur_max = maximum(skipmissing(dur)))
		@ungroup
		@filter(ismissing(version))
	end
	@chain jspsych_data begin
		@filter(prolific_pid == "6658c0f7eacd7bcadeef8a6c")
		@filter(trialphase == "vigour_trial")
		@select(prolific_pid, time_elapsed, trial_duration, trial_presses, trial_number, response_time)
		@mutate(dur = [missing, ~ diff(time_elapsed)...])
	end
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
	
	participants = combine(groupby(data, [:prolific_pid, :record_id, :version, :exp_start_time]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		# :outcomes => extract_PILT_bonus => :PILT_bonus,
		:vigour_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :vigour_bonus,
		# [:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		:trialphase => (x -> sum(skipmissing(x .== "vigour_trial"))) => :n_trial_vigour,
		:trialphase => (x -> sum(skipmissing(x .== "vigour_test"))) => :n_trial_vigour_test,
		# :block => (x -> filter(y -> typeof(y) == Int64, x) |> unique |> length) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
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

	# Function to flatten nested dictionaries
	function flatten_dict(dict, prefix="")
		items = []
		for (key, value) in dict
			new_key = isempty(prefix) ? string(key) : "$(prefix)_$(key)"
			if isa(value, Dict)
				append!(items, flatten_dict(value, new_key))
			else
				push!(items, (new_key, value))
			end
		end
		return items
	end

	# Parse JSON and collect all possible keys
	all_keys = Set()
	parsed_data = []
	for row in eachrow(debrief)
		parsed = JSON.parse(row.debrief_text)
		push!(parsed_data, Dict(flatten_dict(parsed)))
		union!(all_keys, keys(parsed_data[end]))
	end

	# Create DataFrame with all columns, fill missing values with missing
	text_expanded = DataFrame(
		Dict(key => [get(row, key, missing) for row in parsed_data] for key in all_keys)
	)
	
	# hcat together
	return hcat(debrief[!, Not([:debrief_likert, :debrief_text])], likert_expanded, text_expanded)
end

# ╔═╡ Cell order:
# ╠═da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═6eba46dc-855c-47ca-8fa9-8405b9566809
# ╠═bed41c93-2f7e-4f2c-bec4-d5f5cd475e0f
# ╠═e35effd6-5c62-48aa-8932-872c7af50d7b
# ╠═99ca84c3-3df1-448d-b5b6-593328611aa1
# ╠═f221ad44-920c-4b29-a48a-ba6aa529ab45
# ╠═30aebcee-9138-46cf-8dd9-d03f93404517
# ╠═8815162b-6592-44aa-8eb4-8c3d2e2cda59
# ╠═3de841b9-ef50-4cfa-ab5b-dd472ca01d4b
# ╟─431726e1-c06b-40b6-9183-107cc138a5d4
# ╠═6ef2d203-d35e-4480-895f-cf1e553b469a
# ╠═d203faab-d4ea-41b2-985b-33eb8397eecc
# ╠═f47e6aba-00ea-460d-8310-5b24ed7fe336
