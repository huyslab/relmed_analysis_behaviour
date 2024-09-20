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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	nothing
end

# ╔═╡ 51c4f3d4-92e2-40d5-abfc-4438aa438644
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

# ╔═╡ 6eba46dc-855c-47ca-8fa9-8405b9566809
begin
	PLT_data, test_data, vigour_data, jspsych_data = load_pilot2_data()
end

# ╔═╡ d534b22e-8d22-48f5-a6ed-0aa73d5b9fc4
only_observed_test = let

	# Compute EV from PILT
	PLT_data.chosen_stim = replace.(PLT_data.chosenImg, "imgs/" => "")
	
	empirical_EVs = combine(
		groupby(PLT_data, [:session, :prolific_pid, :chosen_stim]),
		:chosenOutcome => mean => :EV
	)

	# Add empirical EVs to test data
	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_left,
			:EV => :empirical_EV_left
		),
		on = [:session, :prolific_pid, :stimulus_left],
		order = :left
	)

	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_right,
			:EV => :empirical_EV_right
		),
		on = [:session, :prolific_pid, :stimulus_right],
		order = :left
	)

	# Compute empirical EV diff
	test_data.empirical_EV_diff = test_data.empirical_EV_right .- 	
		test_data.empirical_EV_left

	# Keep only test trials where stimulus was observed in PILT
	only_observed_test = filter(x -> !ismissing(x.empirical_EV_diff), test_data)

end

# ╔═╡ 9869ebe3-0063-45e6-a59d-52f0607d927a
let
		# Plot distribution of EV difference
	f = Figure(size = (700, 350))

	ax_emp = Axis(
		f[1,1],
		xlabel = "Diff. in empirical EV"
	)

	hist!(ax_emp, 
		only_observed_test.empirical_EV_diff
	)

	ax_exp = Axis(
		f[1,2],
		xlabel = "Diff. in true EV"
	)

	hist!(ax_exp, only_observed_test.EV_diff)

	ax_scatt = Axis(
		f[1,3],
		xlabel = "Diff. in true EV",
		ylabel = "Diff. in empirical EV"
	)

	scatter!(ax_scatt, only_observed_test.EV_diff, only_observed_test.empirical_EV_diff)

	ablines!(ax_scatt, 0., 1., color = :grey, linestyle=:dash)

	f

end

# ╔═╡ 237f096d-9046-49a3-a8c0-f733c89c93bc
let
	only_observed_test
end

# ╔═╡ 16fbf0db-1190-4fb2-a73e-8ef99f1c2999
only_observed_test

# ╔═╡ dba29af6-89c0-4cad-8b5c-b13f6c366982
# Bin and plot choice
function bin_EV_diff_plot(
	f::GridPosition,
	data::AbstractDataFrame;
	group::Union{Nothing, Symbol} = nothing,
	n_bins::Int64 = 5,
	col::Symbol = :empirical_EV_diff, # EV diff column,
	group_label_f::Function = string,
	legend_title = ""
)

	# Copy data to avoid changing origianl DataFrame
	tdata = copy(data)

	# If no grouping is needed
	if isnothing(group)
		tdata[!, :group] .= 1
	else
		rename!(tdata, group => :group)
	end

	# Quantile bin breaks
	EV_bins = quantile(tdata[!, col], 
		range(0, 1, length=n_bins + 1))

	# Bin EV_diff
	tdata.EV_diff_cut = 	
		cut(tdata[!, col], EV_bins, extend = true)

	# Use mean of bin as label
	transform!(
		groupby(tdata, :EV_diff_cut),
		col => mean => :EV_diff_bin
	)

	# Summarize by participant and bin
	choice_EV_sum = combine(
		groupby(tdata, [:prolific_pid, :group, :EV_diff_bin]),
		:right_chosen => mean => :right_chosen
	) |> dropmissing

	# Summarize by bin
	choice_EV_sum = combine(
		groupby(choice_EV_sum, [:group, :EV_diff_bin]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Plot
	ax_diff_choice = Axis(
		f,
		xlabel = "Diff. in EV (£)"
	)

	groups = unique(choice_EV_sum.group)

	for g in groups

		t_sum = filter(x -> x.group == g, choice_EV_sum)

			scatter!(
				ax_diff_choice,
				t_sum.EV_diff_bin,
				t_sum.right_chosen
			)
		
			errorbars!(
				ax_diff_choice,
				t_sum.EV_diff_bin,
				t_sum.right_chosen,
				t_sum.se
			)
	end

	if !isnothing(group)
		Legend(
			f,
			[MarkerElement(marker = :circle, color = Makie.wong_colors()[c]) for c in eachindex(groups)],
			group_label_f.(groups),
			legend_title,
			valign = :top,
			halign = :left,
			tellwidth = false,
			framevisible = false,
			nbanks = 2
		)
	end

	return ax_diff_choice
end

# ╔═╡ 1a507b99-81a7-4a8f-8f55-b3b471956780
let

	f = Figure(size = (700, 300))
	
	bin_EV_diff_plot(f[1,1], only_observed_test)

	bin_EV_diff_plot(f[1,2], only_observed_test; 
		group = :same_block,
		legend_title = "Same original block"
	)

	bin_EV_diff_plot(f[1,3], only_observed_test; 
		group = :same_valence,
		legend_title = "Same original valence"
	)


	f
end

# ╔═╡ fd52ac3c-3d8c-4485-b479-673da579adf0
# Plot PLT accuracy curve
let

	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_data
	)

	ax.xticks = [1, 10, 20, 30]

	f

end

# ╔═╡ ddbc562f-48d5-40dd-969e-48cd3326d3ff
# Plot PLT accuracy curve
let

	PLT_remmaped = copy(PLT_data)

	transform!(
		groupby(PLT_remmaped, [:session, :prolific_pid, :exp_start_time, :block, :stimulus_pair]),
		:trial => (x -> 1:length(x)) => :trial
	)
	
	f = Figure()

	ax = plot_group_accuracy!(
		f[1,1],
		group = :n_pairs,
		PLT_remmaped
	)

	ax.xticks = [1, 10, 20, 30]

	f

end

# ╔═╡ 3caeacb3-e3d2-4f7d-b907-d97fbf831302
let

	# Compute EV from PILT
	PLT_data.chosen_stim = replace.(PLT_data.chosenImg, "imgs/" => "")
	
	empirical_EVs = combine(
		groupby(PLT_data, [:session, :prolific_pid, :chosen_stim]),
		:chosenOutcome => mean => :EV
	)

	# Add empirical EVs to test data
	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_left,
			:EV => :empirical_EV_left
		),
		on = [:session, :prolific_pid, :stimulus_left],
		order = :left
	)

	test_data = leftjoin(
		test_data,
		rename(
			empirical_EVs,
			:chosen_stim => :stimulus_right,
			:EV => :empirical_EV_right
		),
		on = [:session, :prolific_pid, :stimulus_right],
		order = :left
	)

	# Compute empirical EV diff
	test_data.empirical_EV_diff = test_data.empirical_EV_right .- 	
		test_data.empirical_EV_left

	# Keep only test trials where stimulus was observed in PILT
	only_observed_test = filter(x -> !ismissing(x.empirical_EV_diff), test_data)


	# Plot distribution of EV difference
	f = Figure(size = (700, 350))

	ax_emp = Axis(
		f[1,1],
		xlabel = "Diff. in empirical EV"
	)

	hist!(ax_emp, 
		only_observed_test.empirical_EV_diff
	)

	ax_exp = Axis(
		f[1,2],
		xlabel = "Diff. in true EV"
	)

	hist!(ax_exp, only_observed_test.EV_diff)

	ax_scatt = Axis(
		f[1,3],
		xlabel = "Diff. in true EV",
		ylabel = "Diff. in empirical EV"
	)

	scatter!(ax_scatt, only_observed_test.EV_diff, only_observed_test.empirical_EV_diff)

	ablines!(ax_scatt, 0., 1., color = :grey, linestyle=:dash)

	# Bin and plot choice
	function bin_EV_diff_plot(
		f::GridPosition,
		data::AbstractDataFrame;
		n_bins::Int64 = 5
	)
		tdata = copy(data)
		
		EV_bins = quantile(tdata.empirical_EV_diff, 
			range(0, 1, length=n_bins + 1))
		
		tdata.empirical_EV_diff_bin = 	
			cut(only_observed_test.empirical_EV_diff, EV_bins, extend = true)
	
		transform!(
			groupby(tdata, :empirical_EV_diff_bin),
			:empirical_EV_diff => mean => :EV_bin
		)
		
		choice_EV_sum = combine(
			groupby(tdata, [:prolific_pid, :EV_bin]),
			:right_chosen => mean => :right_chosen
		) |> dropmissing
	
		choice_EV_sum = combine(
			groupby(choice_EV_sum, :EV_bin),
			:right_chosen => mean => :right_chosen,
			:right_chosen => sem => :se
		)
	
		ax_diff_choice = Axis(
			f[1,4],
			xlabel = "Diff. in EV"
		)
	
		scatter!(
			ax_diff_choice,
			choice_EV_sum.EV_bin,
			choice_EV_sum.right_chosen
		)
	
		errorbars!(
			ax_diff_choice,
			choice_EV_sum.EV_bin,
			choice_EV_sum.right_chosen,
			choice_EV_sum.se
		)

		return f
	end

	bin_EV_diff_plot(f[1,4], only_observed_test)

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

# ╔═╡ Cell order:
# ╠═da2aa306-75f9-11ef-2592-2be549c73d82
# ╠═51c4f3d4-92e2-40d5-abfc-4438aa438644
# ╠═6eba46dc-855c-47ca-8fa9-8405b9566809
# ╠═d534b22e-8d22-48f5-a6ed-0aa73d5b9fc4
# ╠═9869ebe3-0063-45e6-a59d-52f0607d927a
# ╠═1a507b99-81a7-4a8f-8f55-b3b471956780
# ╠═237f096d-9046-49a3-a8c0-f733c89c93bc
# ╠═16fbf0db-1190-4fb2-a73e-8ef99f1c2999
# ╠═dba29af6-89c0-4cad-8b5c-b13f6c366982
# ╠═4eb3aaed-6028-49a2-9f13-4e915ee2701c
# ╠═fd52ac3c-3d8c-4485-b479-673da579adf0
# ╠═ddbc562f-48d5-40dd-969e-48cd3326d3ff
# ╠═3caeacb3-e3d2-4f7d-b907-d97fbf831302
# ╠═d203faab-d4ea-41b2-985b-33eb8397eecc
# ╠═f47e6aba-00ea-460d-8310-5b24ed7fe336
