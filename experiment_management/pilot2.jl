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
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics
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

# ╔═╡ 720ac162-2113-4078-85e5-289872cb42ac
function compute_optimality(data::AbstractDataFrame)
	
	# Select columns and reduce to task strcuture, which is the same across participants
	optimality = unique(data[!, [:session, :block, :stimulus_pair, :imageLeft, :imageRight, :optimalRight]])

	# Which was the optimal stimulus?
	optimality.optimal = replace.(ifelse.(
		optimality.optimalRight .== 1, 
		optimality.imageRight, 
		optimality.imageLeft
	), "imgs/" => "")

	# Which was the suboptimal stimulus?
	optimality.suboptimal = replace.(ifelse.(
		optimality.optimalRight .== 0, 
		optimality.imageRight, 
		optimality.imageLeft
	), "imgs/" => "")

	# Remove double appearances (right left permutation)
	optimality = unique(optimality[!, [:session, :block, :stimulus_pair, 
		:optimal, :suboptimal]])

	# Wide to long
	optimality = DataFrame(
		stimulus = vcat(optimality.optimal, optimality.suboptimal),
		optimal = vcat(fill(true, nrow(optimality)), fill(false, nrow(optimality)))
	)

	return optimality
end

# ╔═╡ 8b212920-6363-4161-96ed-d2060e4822b9
function stimulus_magnitude()

	task = DataFrame(CSV.File("./results/pilot2.csv"))

	outcomes = filter(x -> x.feedback_common, task)

	outcomes = vcat(
		rename(
			outcomes[!, [:stimulus_right, :feedback_right]],
			:stimulus_right => :stimulus,
			:feedback_right => :feedback
		),
		rename(
			outcomes[!, [:stimulus_left, :feedback_left]],
			:stimulus_left => :stimulus,
			:feedback_left => :feedback
		)
	)

	outcomes = combine(
		groupby(outcomes, :stimulus),
		:feedback => (x -> mean(unique(x))) => :feedback
	)

	return outcomes

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

	# Coarse stimulus magnitude
	magnitudes = stimulus_magnitude()

	# Add to test data
	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			magnitudes,
			:stimulus => :stimulus_left,
			:feedback => :magnitude_left
		),
		on = :stimulus_left,
		order = :left
	)

	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			magnitudes,
			:stimulus => :stimulus_right,
			:feedback => :magnitude_right
		),
		on = :stimulus_right,
		order = :left
	)

	# Compute optimality of each stimulus
	optimality = compute_optimality(PLT_data)

	# Add to test data
	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			optimality,
			:stimulus => :stimulus_left,
			:optimal => :optimal_left
		),
		on = :stimulus_left,
		order = :left
	)

	only_observed_test = leftjoin(
		only_observed_test,
		rename(
			optimality,
			:stimulus => :stimulus_right,
			:optimal => :optimal_right
		),
		on = :stimulus_right,
		order = :left
	)

	# Compute optimality categories
	only_observed_test.optimcat = ifelse.(
		only_observed_test.optimal_right .&& only_observed_test.optimal_left,
		fill("both", nrow(only_observed_test)),
		ifelse.(
			only_observed_test.optimal_right,
			fill("right", nrow(only_observed_test)),
			ifelse.(
				only_observed_test.optimal_left,
				fill("left", nrow(only_observed_test)),
				fill("none", nrow(only_observed_test))
			)
		)
	)

	only_observed_test

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

# ╔═╡ fa7e77e6-0008-4249-9672-e4f867aacd9c
only_observed_test

# ╔═╡ 2ce13a6e-ec58-4c7c-bb08-88881cad8eaa
let

	f = Figure()

	# Drop skipped trials
	dat = dropmissing(
		only_observed_test[!, 
			[:prolific_pid, :magnitude_left, :magnitude_right, :optimal_left, :optimal_right, :right_chosen]])

	# Plot by valence ----------------------

	# Compute valence from magnitude
	transform!(
		dat,
		:magnitude_left => ByRow(sign) => :valence_left,
		:magnitude_right => ByRow(sign) => :valence_right
	)

	# Average by participant and valence
	val = combine(
		groupby(dat, [:prolific_pid, :valence_right, :valence_left]),
		:right_chosen => mean => :right_chosen
	)

	# Average by valence
	val = combine(
		groupby(val, [:valence_right, :valence_left]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Create one valence variable
	val.type = 
		join.(
			eachrow(hcat(
				ifelse.(
					val.valence_left .> 0,
					fill("P", nrow(val)),
					fill("N", nrow(val))
				),
				ifelse.(
					val.valence_right .> 0,
					fill("P", nrow(val)),
					fill("N", nrow(val))
				)
			))
		)

	# Plot bars
	bar = data(val) * mapping(:type, :right_chosen) * visual(BarPlot)

	# Plot error bars
	err = data(val) * mapping(:type, :right_chosen, :se => (x -> x*2)) * visual(Errorbars)

	# Plot chance
	hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)

	# Put together
	draw!(f[1,1], bar + err + hline; axis = (; xlabel = "Valence", ylabel = "Prop. right chosen"))

	# Plot by optimality
	# Average by participant and valence
	opt = combine(
		groupby(dat, [:prolific_pid, :optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen
	)

	# Average by valence
	opt = combine(
		groupby(opt, [:optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

		# Create one valence variable
	opt.type = 
		join.(
			eachrow(hcat(
				ifelse.(
					opt.optimal_left,
					fill("O", nrow(val)),
					fill("S", nrow(val))
				),
				ifelse.(
					opt.optimal_right,
					fill("O", nrow(val)),
					fill("S", nrow(val))
				)
			))
		)

	# Plot bars
	let
		bar = data(opt) * mapping(:type, :right_chosen) * visual(BarPlot)
	
		# Plot error bars
		err = data(opt) * mapping(:type, :right_chosen, :se => (x -> x*2)) * visual(Errorbars)
	
		# Plot chance
		hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)
	
		# Put together
		draw!(f[1,2], bar + err + hline; axis = (; xlabel = "Optimality", ylabel = "Prop. right chosen"))
	end

	# Plot by valence and optimality
	# Average by participant, valence, and optimality
	val_opt = combine(
		groupby(dat, 
			[:prolific_pid, 
				:valence_right, :valence_left, 
				:optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen
	)

	# Average by valence and optimality
	val_opt = combine(
		groupby(val_opt,
			[:valence_right, :valence_left, :optimal_right, :optimal_left]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Create one variable
	val_opt.val_type = 
		join.(
			eachrow(hcat(
				ifelse.(
					val_opt.valence_left .> 0,
					fill("P", nrow(val_opt)),
					fill("N", nrow(val_opt))
				),
				ifelse.(
					val_opt.valence_right .> 0,
					fill("P", nrow(val_opt)),
					fill("N", nrow(val_opt))
				)
			))
		)

	val_opt.opt_type = 
		join.(
			eachrow(hcat(
				ifelse.(
					val_opt.optimal_left,
					fill("O", nrow(val_opt)),
					fill("S", nrow(val_opt))
				),
				ifelse.(
					val_opt.optimal_right,
					fill("O", nrow(val_opt)),
					fill("S", nrow(val_opt))
				)
			))
		)


	val_opt.type = join.(eachrow(hcat(val_opt.opt_type, val_opt.val_type)), "\n")

	let
		bar = data(val_opt) * mapping(:type, :right_chosen) * visual(BarPlot)
	
		# Plot error bars
		err = data(val_opt) * mapping(:type, :right_chosen, :se => (x -> x*2)) * visual(Errorbars)
	
		# Plot chance
		hline = mapping([0.5]) * visual(HLines; color = :grey, linestyle = :dash)
	
		# Put together
		draw!(f[2,:], bar + err + hline; axis = (; xlabel = "Stimuli", ylabel = "Prop. right chosen"))
	end
	
	f
end

# ╔═╡ 63acbf9a-33f8-47e5-a85b-95c8cd57b12b
let
	dat = dropmissing(only_observed_test[!, [:prolific_pid, :magnitude_left, :magnitude_right, :right_chosen]])

	dat.magnitude_low = minimum(hcat(dat.magnitude_right, dat.magnitude_left), dims = 2) |> vec

	dat.magnitude_high = maximum(hcat(dat.magnitude_right, dat.magnitude_left), dims = 2) |> vec

	dat.high_chosen = ifelse.(
		dat.magnitude_right .== dat.magnitude_high,
		dat.right_chosen,
		.!dat.right_chosen
	)
	
	dat_sum = combine(
		groupby(dat, [:prolific_pid, :magnitude_low, :magnitude_high]),
		:high_chosen => mean => :high_chosen,
		:high_chosen => length => :n
	)

	dat_sum = combine(
		groupby(dat_sum, [:magnitude_low, :magnitude_high]),
		:high_chosen => mean => :high_chosen,
		:high_chosen => sem => :se,
		:n => median => :n
	)

	filter!(x -> !(x.magnitude_low == x.magnitude_high), dat_sum)

	sort!(dat_sum, [:magnitude_low, :magnitude_high])

	dat_sum.high_optimal = (x -> x in [-0.255, -0.01, 0.75, 1.]).(dat_sum.magnitude_high)

	plt = data(dat_sum) * 
		visual(ScatterLines) * 
		mapping(:magnitude_low => nonnumeric, :high_chosen, 
			markersize = :n, layout = :magnitude_high => nonnumeric, 
			color = :high_optimal => nonnumeric)

	err = data(dat_sum) * 
		mapping(
			:magnitude_low => nonnumeric, 
			:high_chosen, 
			:se, 
			color = :high_optimal => nonnumeric,
			layout = :magnitude_high => nonnumeric) * 
		visual(Errorbars)
	hline = mapping(0.5) * visual(HLines, color = :grey, linestyle = :dash)

	spans = DataFrame(
		low = (1:7) .- 0.5,
		high = (1:7) .+ 0.5,
		optimal = [false, false, true, true, false, false, true]
	)

	color = [:red, :blue]

	vspans = data(spans) * mapping(:low, :high, color = :optimal => nonnumeric => AlgebraOfGraphics.scale(:secondary)) * visual(VSpan)


	draw(vspans + plt + err + hline, 
		scales(
			secondary = (; palette = [(:red, 0.2), (:green, 0.2)]), 
			Color = (; palette = [:red, :green])); 
		legend = (; show = false), 
		axis = (; xticklabelrotation=45.0, 
			xlabel = "Low magnitude",
			ylabel = "Prop. chosen high magnitude"
		)
	)
end

# ╔═╡ d57501e1-3574-4954-88e5-8cccd9a9584b
describe(only_observed_test)

# ╔═╡ eb8308f3-f442-467d-933d-5273de71d1f6
function bin_sum_EV(
	data::DataFrame;
	group::Union{Nothing, Symbol} = nothing,
	n_bins::Int64 = 5,
	col::Symbol = :empirical_EV_diff, # x axis data column,
	bin_group::Union{Nothing, Int64} = nothing
)
	# Copy data to avoid changing origianl DataFrame
	tdata = copy(data)

	# If no grouping is needed
	if isnothing(group)
		tdata[!, :group] .= 1
	else
		if !isnothing(bin_group)
			
			# Quantile bin breaks
			group_bins = quantile(tdata[!, col], 
				range(0, 1, length=bin_group + 1))

			# Bin group
			tdata.EV_group_cut = 	
				cut(tdata[!, group], group_bins, extend = true)
		
			# Use mean of bin as label
			transform!(
				groupby(tdata, :EV_group_cut),
				group => mean => :group
			)
		else
			rename!(tdata, group => :group)
		end
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


end

# ╔═╡ dba29af6-89c0-4cad-8b5c-b13f6c366982
# Bin and plot choice
function bin_EV_plot(
	f::GridPosition,
	data::AbstractDataFrame;
	group::Union{Nothing, Symbol} = nothing,
	n_bins::Int64 = 5,
	col::Symbol = :empirical_EV_diff, # x axis data column,
	group_label_f::Function = string,
	legend_title = "",
	colors::AbstractVector = Makie.wong_colors(),
	bin_group::Union{Nothing, Int64} = nothing
)

	choice_EV_sum = bin_sum_EV(
		data,
		group = group,
		n_bins = n_bins,
		col = col,
		bin_group = bin_group
	)
	
	# Plot
	ax_diff_choice = Axis(
		f,
		xlabel = "Diff. in EV (£)"
	)

	groups = sort(unique(choice_EV_sum.group))

	for g in groups

		t_sum = filter(x -> x.group == g, choice_EV_sum)

			scatter!(
				ax_diff_choice,
				t_sum.EV_diff_bin,
				t_sum.right_chosen,
				colormap = :viridis
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
			[MarkerElement(marker = :circle, color = colors[c]) for c in eachindex(groups)],
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
	
	bin_EV_plot(f[1,1], only_observed_test)

	bin_EV_plot(f[1,2], only_observed_test; 
		group = :block,
		legend_title = "Block"
	)

	bin_EV_plot(f[1,3], only_observed_test; 
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
# ╠═720ac162-2113-4078-85e5-289872cb42ac
# ╠═8b212920-6363-4161-96ed-d2060e4822b9
# ╠═d534b22e-8d22-48f5-a6ed-0aa73d5b9fc4
# ╠═9869ebe3-0063-45e6-a59d-52f0607d927a
# ╠═1a507b99-81a7-4a8f-8f55-b3b471956780
# ╠═fa7e77e6-0008-4249-9672-e4f867aacd9c
# ╠═2ce13a6e-ec58-4c7c-bb08-88881cad8eaa
# ╠═63acbf9a-33f8-47e5-a85b-95c8cd57b12b
# ╠═d57501e1-3574-4954-88e5-8cccd9a9584b
# ╠═eb8308f3-f442-467d-933d-5273de71d1f6
# ╠═dba29af6-89c0-4cad-8b5c-b13f6c366982
# ╠═4eb3aaed-6028-49a2-9f13-4e915ee2701c
# ╠═fd52ac3c-3d8c-4485-b479-673da579adf0
# ╠═ddbc562f-48d5-40dd-969e-48cd3326d3ff
# ╠═d203faab-d4ea-41b2-985b-33eb8397eecc
# ╠═f47e6aba-00ea-460d-8310-5b24ed7fe336
