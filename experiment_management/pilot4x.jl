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

# ╔═╡ eafd04fa-05ab-4f29-921a-63890e8c83a0
function load_pilot4x_data(; force_download = false)
	datafile = "data/pilot4.x.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot4.x"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PLT_data = prepare_PLT_data(jspsych_data)

	# Extract post-PILT test
	test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	return PLT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, reversal_data, jspsych_data
end


# ╔═╡ 57ca3929-faa6-4a95-9e4d-6c1add13b121
PLT_data, test_data, vigour_data, post_vigour_test_data, pit_data, reversal_data, jspsych_data = load_pilot4x_data()

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
		# [:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
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
summarize_participation(jspsych_data)

# ╔═╡ Cell order:
# ╠═baba7ea8-9069-11ef-2bba-89fb74ddc46b
# ╠═eafd04fa-05ab-4f29-921a-63890e8c83a0
# ╠═57ca3929-faa6-4a95-9e4d-6c1add13b121
# ╠═28e7943a-37eb-49f0-add0-382ed02c6f68
# ╠═8ed5b9b8-867c-439b-8fcb-4303203ae95e
# ╠═ecd4060f-21a1-4c98-93c6-c33211c7a485
