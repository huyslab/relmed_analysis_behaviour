### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ e953ab1e-2f64-11f0-287e-afd1a5effa1f
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

# ╔═╡ da326055-094a-4b3a-8a0c-dbb87366b3ae
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

# ╔═╡ 87317512-19fe-4a5e-b7d2-5b4161c9faf1
ENV["pilot9_REDCap_token"]="92296069CE80017E9531895B56DBE494"

# ╔═╡ ce32edbe-2883-4b02-ad66-49228b6c2bac
function prepare_test_data(df::DataFrame; task::String = "pilt")

	# Select rows
	test_data = filter(x -> (x.trial_type == "PILT") && (x.trialphase == "$(task)_test"), df)

	# Select columns
	test_data = test_data[:, Not(map(col -> all(ismissing, col), eachcol(test_data)))]

	# Change all block names to same type
	test_data.block = string.(test_data.block)

	# Sort
	sort!(test_data, [:participant_id, :session, :block, :trial])

	return test_data

end


# ╔═╡ 9eb71529-0dc7-4951-9e2e-96ec95f4df89
function load_pilot9_data(; force_download = false)
	datafile = "data/pilot9.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot9"; file_field = "data", record_id_field = "record_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "sitting_start_time")

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Seperate out PILT
	filter!(x -> x.trialphase == "pilt", PILT_data)
	
	# Extract post-PILT test
	test_data = prepare_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 
			
	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Extract max press rate data
	# max_press_data = prepare_max_press_data(jspsych_data)

	# Extract control data
	# control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return PILT_data, test_data, vigour_data, PIT_data, jspsych_data
end


# ╔═╡ b00006c6-ef32-45d3-b5a2-d6614f39bc31
begin
	PILT_data, test_data, vigour_data, PIT_data, jspsych_data = load_pilot9_data()
	nothing
end

# ╔═╡ 781c45d2-d4b6-463f-87bf-a01db1fc0372
md"""# PILT"""

# ╔═╡ 34e147de-0af9-45a6-aa56-785d7d071153
let
	@assert all((x -> x in ["right", "left", "noresp"]).(unique(PILT_data.response))) "Unexpected values in response"
	
	@assert all(PILT_data.chosen_feedback .== ifelse.(
		PILT_data.response .== "right",
		PILT_data.feedback_right,
		ifelse.(
			PILT_data.response .== "left",
			PILT_data.feedback_left,
			minimum.(hcat.(PILT_data.feedback_left, PILT_data.feedback_right))
		)
	)) "`chosen_feedback` doens't match feedback and choice"


end

# ╔═╡ aaf73642-aaec-40a9-95f8-f7f0f25095c7
let
	# Clean data
	PILT_data_clean = exclude_PLT_sessions(PILT_data, required_n_blocks = 21)
	PILT_data_clean = filter(x -> x.response != "noresp", PILT_data_clean)

	# Sumarrize by participant and trial
	acc_curve = combine(
		groupby(PILT_data_clean, [:prolific_pid, :trial]),
		:response_optimal => mean => :acc
	)

	sort!(acc_curve, [:prolific_pid, :trial])

	# Summarize by trial
	acc_curve_sum = combine(
		groupby(acc_curve, :trial),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Plot
	mp = (data(acc_curve) * mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice",
		group = :prolific_pid,
		color = :prolific_pid
	) * visual(Lines, linewidth = 1, alpha = 0.7)) +
	(data(acc_curve_sum) * 
	mapping(
		:trial => "Trial #",
		:acc => "Prop. optimal choice"
	) * visual(Lines, linewidth = 4))
	
	
	draw(mp; legend = (; show = false))
end

# ╔═╡ 5373906f-51fb-43aa-91f7-20656785a386
let
	test_data_clean = filter(x -> x.response != "no_resp", test_data)
	
	test_data_clean.EV_diff = test_data_clean.EV_right .- test_data_clean.EV_left

	test_sum = combine(
		groupby(test_data_clean, :EV_diff),
		:response => (x -> mean(x .== "right")) => :right_chosen
	)

	mp = data(test_sum) * mapping(:EV_diff, :right_chosen) * visual(Scatter)

	draw(mp)
end

# ╔═╡ 0dbe2eb5-e83c-4d26-acb1-1151a698a873
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

# ╔═╡ 5725d5d4-bb41-4cf1-8256-a2e3132a677b
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

# ╔═╡ 00d58d1e-9280-4c7e-8399-675db7abb89b
begin
	p_sum = summarize_participation(jspsych_data)
end

# ╔═╡ Cell order:
# ╠═e953ab1e-2f64-11f0-287e-afd1a5effa1f
# ╠═da326055-094a-4b3a-8a0c-dbb87366b3ae
# ╠═9eb71529-0dc7-4951-9e2e-96ec95f4df89
# ╠═87317512-19fe-4a5e-b7d2-5b4161c9faf1
# ╠═ce32edbe-2883-4b02-ad66-49228b6c2bac
# ╠═b00006c6-ef32-45d3-b5a2-d6614f39bc31
# ╠═00d58d1e-9280-4c7e-8399-675db7abb89b
# ╟─781c45d2-d4b6-463f-87bf-a01db1fc0372
# ╠═34e147de-0af9-45a6-aa56-785d7d071153
# ╠═aaf73642-aaec-40a9-95f8-f7f0f25095c7
# ╠═5373906f-51fb-43aa-91f7-20656785a386
# ╠═5725d5d4-bb41-4cf1-8256-a2e3132a677b
# ╠═0dbe2eb5-e83c-4d26-acb1-1151a698a873
