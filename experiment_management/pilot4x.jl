### A Pluto.jl notebook ###
# v0.20.3

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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, Tidier
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 57ca3929-faa6-4a95-9e4d-6c1add13b121
PLT_data, test_data, vigour_data, post_vigour_test_data, pit_data, reversal_data, jspsych_data = load_pilot4x_data(force_download=true);

# ╔═╡ b17f06f1-31bc-4cc0-af64-aa8265d7d796
jspsych_data |>
	y -> filter(x -> x.prolific_pid == "667c11fddf1df5ac03db2839", y)

# ╔═╡ d41490e4-2d3b-480e-a627-663874de0e93
(jspsych_data) |>
	x -> filter(y -> y.prolific_pid == "667c11fddf1df5ac03db2839", x) |>
	x -> filter(y -> !ismissing(y.trialphase) && y.trialphase == "task", x)

# ╔═╡ 28e7943a-37eb-49f0-add0-382ed02c6f68
"""
    extract_debrief_responses(data::DataFrame) -> DataFrame

Extracts and processes debrief responses from the experimental data. It filters for debrief trials, then parses and expands JSON-formatted responses from multiple acceptability and debrief columns into separate columns for each question.

# Arguments
- `data::DataFrame`: The raw experimental data containing participants' trial outcomes and responses, including debrief information.

# Returns
- A DataFrame with participants' debrief responses. The acceptability and debrief responses are parsed from JSON and expanded into separate columns.
"""
function extract_debrief_responses(data::DataFrame)
    # Select trials
    debrief = filter(x -> !ismissing(x.trialphase) && occursin(r"^(acceptability_|debrief_)", x.trialphase), data)

    # Select variables
    select!(debrief, [:prolific_pid, :exp_start_time, :trialphase, :response])
    
    # Long to wide
    debrief = unstack(
        debrief,
        [:prolific_pid, :exp_start_time],
        :trialphase,
        :response
    )

    # Get response columns (excluding identifier columns)
    response_columns = setdiff(names(debrief), ["prolific_pid", "exp_start_time"])
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

    # Initialize DataFrames for expanded columns
    expanded_dfs = []

    # Process each response column
    for col in response_columns
        # Parse JSON and create expanded DataFrame for each column
        try
            parsed_data = []
            all_keys = Set()

            for row in eachrow(debrief)
                if ismissing(row[col])
                    push!(parsed_data, Dict())
                    continue
                end
                
                parsed = JSON.parse(row[col])
                if isa(parsed, Dict)
                    flattened = Dict(flatten_dict(parsed))
                    push!(parsed_data, flattened)
                    union!(all_keys, keys(flattened))
                else
                    # Handle non-dictionary JSON (e.g., single values)
                    push!(parsed_data, Dict("value" => parsed))
                    push!(all_keys, "value")
                end
            end

            # Create DataFrame with all columns, fill missing values with missing
            if !isempty(all_keys)
                expanded_df = DataFrame(
                    Dict(key => [get(row, key, missing) for row in parsed_data] for key in all_keys)
                )
                
                # Rename columns to include original column name as prefix, without duplication
                rename!(expanded_df, [name => "$(col)_$(name)" for name in names(expanded_df)])
                
                push!(expanded_dfs, expanded_df)
            end
        catch e
            @warn "Failed to process column $col: $e"
            continue
        end
    end

    # Combine all DataFrames
    if isempty(expanded_dfs)
        return debrief
    else
        return hcat(
            debrief[!, [:prolific_pid, :exp_start_time]], 
            expanded_dfs..., 
            makeunique=true
        )
    end
end

# ╔═╡ 8ed5b9b8-867c-439b-8fcb-4303203ae95e
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
		:outcomes => extract_PILT_bonus => :PILT_bonus,
		:vigour_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :vigour_bonus,
		:PIT_bonus => (x -> all(ismissing.(x)) ? missing : (filter(y -> !ismissing(y), unique(x))[1])) => :PIT_bonus,
		[:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		:trialphase => (x -> sum(skipmissing(x .== "PILT_test"))) => :n_trial_pilt_test,
		:trialphase => (x -> sum(skipmissing(x .== "vigour_trial"))) => :n_trial_vigour,
		:trialphase => (x -> sum(skipmissing(x .== "vigour_test"))) => :n_trial_vigour_test,
		:trialphase => (x -> sum(skipmissing(x .== "pit_trial"))) => :n_trial_pit,
		# :block => (x -> filter(y -> typeof(y) == Int64, x) |> unique |> length) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	debrief = extract_debrief_responses(data)

	participants = leftjoin(participants, debrief, 
		on = [:prolific_pid, :exp_start_time])

	return participants
end

# ╔═╡ ecd4060f-21a1-4c98-93c6-c33211c7a485
summarize_participation(jspsych_data) |>
	filter(x -> x.version in ["4.2"] && !ismissing(x.finished))

# ╔═╡ c1b6ca2d-65ca-4aaf-a226-826fb473657c
let
	bonus_df = summarize_participation(jspsych_data) |>
		filter(x -> x.version in ["4.2"] && !ismissing(x.finished))
	bonus_df.total_bonus = round.(sum(eachcol(select(bonus_df, r"_bonus")));digits=2)
	
	for row in eachrow(select(bonus_df, :prolific_pid => :"Participant id", :total_bonus))
		println(join(row, ","))
	end
end

# ╔═╡ ca4bd168-2a58-4bfc-9108-be358683bd78
begin
	sub_list = CSV.read("data/prolific_participant_info/prolific_export_671e1fd663e930a72f26257b.csv", DataFrame)
	finished_df = filter(x -> !ismissing(x.finished) && x.finished && !ismissing(x.vigour_bonus), summarize_participation(jspsych_data)) |>
	x -> select(x, :prolific_pid, :vigour_bonus) |>
	x -> subset(x, :vigour_bonus => ByRow(x -> x > 0))
	finished_df = semijoin(finished_df, sub_list, on = [:prolific_pid => Symbol("Participant id")])
	for row in eachrow(finished_df)
		println(join(row, ","))
	end
end

# ╔═╡ 5e329eef-81f1-4c7a-a26a-30d6200095b3
let
	finished = summarize_participation(jspsych_data) |>
		filter(x -> x.version in ["4.3"] && !ismissing(x.finished)) |>
		x -> select(x, :prolific_pid)
	for row in eachrow(finished)
		println(join(row, ","))
	end
end

# ╔═╡ 13326504-4b52-4f4e-8c48-b2ae0fb2febd
summarize_participation(jspsych_data) |>
	x -> filter(y -> y.prolific_pid == "5d62356508ff870001878e92", x)

# ╔═╡ 3463a1c9-5f44-40da-a60a-51d6a0137ad9
summarize_participation(jspsych_data) |>
	filter(x -> x.version in ["4.1"] && !ismissing(x.finished))

# ╔═╡ Cell order:
# ╠═baba7ea8-9069-11ef-2bba-89fb74ddc46b
# ╠═57ca3929-faa6-4a95-9e4d-6c1add13b121
# ╠═ecd4060f-21a1-4c98-93c6-c33211c7a485
# ╠═c1b6ca2d-65ca-4aaf-a226-826fb473657c
# ╠═ca4bd168-2a58-4bfc-9108-be358683bd78
# ╠═5e329eef-81f1-4c7a-a26a-30d6200095b3
# ╠═b17f06f1-31bc-4cc0-af64-aa8265d7d796
# ╠═13326504-4b52-4f4e-8c48-b2ae0fb2febd
# ╠═3463a1c9-5f44-40da-a60a-51d6a0137ad9
# ╠═d41490e4-2d3b-480e-a627-663874de0e93
# ╠═8ed5b9b8-867c-439b-8fcb-4303203ae95e
# ╟─28e7943a-37eb-49f0-add0-382ed02c6f68
