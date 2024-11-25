### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ fce1a1b4-a5cf-11ef-0068-c7282cd862f0
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates, SHA
	import OpenScienceFramework as OSF
	using LogExpFunctions: logistic, logit
	using Tidier
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("plotting_utils.jl")
	include("osf_utils.jl")
	nothing
end

# ╔═╡ 0978c5a9-b488-44f0-8a6c-9d3e51da4c3a
set_theme!(theme_minimal();font = "Helvetica",
		fontsize = 16)

# ╔═╡ aae6bde5-84f8-4ecd-8ec9-bc1cf7f92583
# Set up saving to OSF
begin
	osf_folder = "Workshop figures/Acceptability"
	proj = setup_osf("Task development")
end

# ╔═╡ caa2e442-9eff-4c4d-a6f2-b09cfccf2357
begin
	# Load data
	PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data, 
		reversal_data, jspsych_data = load_pilot6_data()
	nothing
end

# ╔═╡ 23f4c513-c4af-4bde-adac-8cdd88e48333
md"""
## Acceptability ratings
"""

# ╔═╡ d71d62c5-f617-4a5a-a27c-8a9820347b76
md"""
## Time elapsed on each task
"""

# ╔═╡ 15950e90-11b7-45fb-9507-8f5e805d6ce4
unique(jspsych_data.trialphase)

# ╔═╡ f51aa34a-5501-41f2-b12f-4340d0cdaf26
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

# ╔═╡ 2481764a-be2c-413b-bd48-e460c00fe2ff
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
			return maximum([0., bonus])
		end

	end
	
	participants = combine(groupby(data, [:prolific_pid, :record_id, :exp_start_time, :session]),
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

# ╔═╡ 8191dbc5-a8dd-49b0-bd30-58c1d4744f74
begin
	p_sum = summarize_participation(jspsych_data)
	CSV.write("results/workshop/acceptability.csv", p_sum)
	@info "# Valid data samples: $(sum(skipmissing(p_sum.finished)))"
end

# ╔═╡ b083eaa9-9b09-423c-bc32-9c5f17a91391
accept_data = @chain p_sum begin
	@filter(!ismissing(finished) & finished)
	@select(prolific_pid, session, matches("_(difficulty|clear|enjoy)"))
	@pivot_longer(-(prolific_pid:session), names_to = "item", values_to = "score")
	@separate(item, [task, question], "_")
end

# ╔═╡ 3f94af37-bc0d-4bd9-ba4c-5fa244e2b3a3
let
	figs = []

	tasks = unique(accept_data.task)
	
	for (i, t) in enumerate(tasks)
		
		fig = Figure(size = (41.47, 14.78) .* 36 ./ 2.54)

		pdata = copy(accept_data)

		transform!(pdata, 
			:task => (x -> x .== t) => :highlight,
			:score => (x -> x .+ 1) => :score
		)

		pdata.task = CategoricalArray(
			pdata.task, 
			levels = vcat(filter(x -> x != t, tasks), [t])
		)
		
		p = 
			data(pdata) * 
			(
				mapping(
					:task => "", 
					:score => "",
					color = :highlight,
					col = :question
				) * 
				visual(BoxPlot, orientation = :horizontal, outliercolor = :white) 
			)
		
		draw!(
			fig[1,1], 
			p, 
			scales(
				Color = (; palette = [:grey, Makie.wong_colors()[1]]),
				Y = (; 
					categories = reverse([
						"pilt" => "PILT",
						"vigour" => "Vigour",
						"pit" => "PIT",
						"wm" => "WM",
						"reversal" => "Reversal"
					])
				),
				Col = (;
					categories = [
						"clear" => "Clarity",
						"difficulty" => "Difficulty",
						"enjoy" => "Enjoyment"
					]
				)
			); 
			facet=(; linkxaxes=:none)
		)

		save("results/workshop/acceptability_$t.png", fig, pt_per_unit = 1)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 565e9389-f251-4df0-9e4d-8fab4e009611
let
	figs = []

	tasks = unique(accept_data.task)
	
	for (i, t) in enumerate(tasks)
		
		fig = Figure(size = (700, 200))

		pdata = combine(
			groupby(accept_data, [:task, :question]),
			:score => (x -> mean(x .+ 1)) => :score,
			:score => (x -> sem(x .+ 1)) => :se,
			:score => (x -> mean(x .+ 1) - lb(x .+ 1)) => :lb,
			:score => (x -> ub(x .+ 1) - mean(x .+ 1)) => :ub,
		)

		transform!(pdata, :task => (x -> x .== t) => :highlight)

		pdata.task = CategoricalArray(
			pdata.task, 
			levels = vcat(filter(x -> x != t, tasks), [t])
		)
		
		p = 
			data(pdata) * 
			(
				mapping(
					:score => "",
					:task => "", 
					:lb,
					:ub; 
					color = :highlight,
					col = :question
				) * 
				visual(Errorbars, direction = :x) +
				mapping(
					:score => "",
					:task => "", 
					color = :highlight,
					col = :question
				) * 
				visual(Scatter)
			)
		
		draw!(
			fig[1,1], 
			p, 
			scales(
				Color = (; palette = [:grey, Makie.wong_colors()[1]]),
				Y = (; 
					palette = [1, 2, 3, 4, 6], 
					categories = reverse([
						"pilt" => "PILT",
						"vigour" => "Vigour",
						"pit" => "PIT",
						"wm" => "WM",
						"reversal" => "Reversal"
					])
				),
				Col = (;
					categories = [
						"clear" => "Clarity",
						"difficulty" => "Difficulty",
						"enjoy" => "Enjoyment"
					]
				)
			); 
			facet=(; linkxaxes=:none),
			axis = (; limits = (1, 5, nothing, nothing))
		)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ 864f7f16-7408-441c-89bf-c0e6894ffce4
let
	figs = []

	tasks = unique(accept_data.task)

	pdata = combine(
			groupby(accept_data, [:task, :question]),
			:score => (x -> mean(x .+ 1)) => :score,
			:score => (x -> sem(x .+ 1)) => :se,
			:score => (x -> mean(x .+ 1) - lb(x .+ 1)) => :lb,
			:score => (x -> ub(x .+ 1) - mean(x .+ 1)) => :ub,
		)
	
	for (i, t) in enumerate(tasks)
		
		fig = Figure(size = (41.47, 14.78) .* 36 ./ 2.54)

		transform!(pdata, :task => (x -> x .== t) => :highlight)

		pdata.task = CategoricalArray(
			pdata.task, 
			levels = vcat(filter(x -> x != t, tasks), [t])
		)
		
		p = 
			data(pdata) * 
			(
				mapping(
					:score => "",
					:task => "", 
					:se; 
					color = :highlight,
					col = :question
				) * 
				visual(Errorbars, direction = :x) +
				mapping(
					:score => "",
					:task => "", 
					color = :highlight,
					col = :question
				) * 
				visual(Scatter)
			)
		
		draw!(
			fig[1,1], 
			p, 
			scales(
				Color = (; palette = [:grey, Makie.wong_colors()[1]]),
				Y = (; 
					categories = reverse([
						"pilt" => "PILT",
						"vigour" => "Vigour",
						"pit" => "PIT",
						"wm" => "WM",
						"reversal" => "Reversal"
					])
				),
				Col = (;
					categories = [
						"clear" => "Clarity",
						"difficulty" => "Difficulty",
						"enjoy" => "Enjoyment"
					]
				)
			); 
			facet=(; linkxaxes=:none),
			axis = (; limits = (1, 5, nothing, nothing))
		)
		
		push!(figs, fig)
	end
	figs
end

# ╔═╡ e2711875-18a0-4c16-835a-9e12d8f6316e
durations = let
	finishers = filter(x -> (!ismissing(x.finished)) && x.finished, p_sum)
	finishers = [(r.prolific_pid, r.session, r.exp_start_time) 
		for r in eachrow(finishers)]

	gb = groupby(
		filter(
			r -> (r.prolific_pid, r.session, r.exp_start_time) in finishers,
			jspsych_data
		), 
		[:prolific_pid, :session, :exp_start_time]
	)

	# PILT duration
	PILT = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "instruction"))-1]) 
				=> :start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "pre_debrief_instructions"))]) => :end_time
	)

	PILT.duration = (PILT.end_time .- PILT.start_time) ./ 60000

	PILT[!, :task] .= "PILT"

	# PILT net duration
	PILT_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "instruction_quiz"))]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "pre_debrief_instructions"))]) 
				=> :end_time
	)

	PILT_net.duration = (PILT_net.end_time .- PILT_net.start_time) ./ 60000

	PILT_net[!, :task] .= "PILT net"


	# Vigour duration
	vigour = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "vigour_instructions"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "vigour_trial"))]) => :end_time
	)

	vigour.duration = (vigour.end_time .- vigour.start_time) ./ 60000

	vigour[!, :task] .= "Vigour"

	# Vigour net duration
	vigour_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "vigour_trial"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "vigour_trial"))]) => :end_time
	)

	vigour_net.duration = (vigour_net.end_time .- vigour_net.start_time) ./ 60000

	vigour_net[!, :task] .= "Vigour net"


	# PIT duration
	PIT = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "pit_instructions"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "pit_trial"))]) => :end_time
	)

	PIT.duration = (PIT.end_time .- PIT.start_time) ./ 60000

	PIT[!, :task] .= "PIT"

	# PIT net duration
	PIT_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "pit_instructions"))]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "pit_trial"))]) => :end_time
	)

	PIT_net.duration = (PIT_net.end_time .- PIT_net.start_time) ./ 60000

	PIT_net[!, :task] .= "PIT net"


	# Test duration
	test = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "post-PILT_test_instructions"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "PILT_test"))]) => :end_time
	)

	test.duration = (test.end_time .- test.start_time) ./ 60000

	test[!, :task] .= "Post-PILT test"

	# Test net duration
	test_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "post-PILT_test_instructions"))]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "PILT_test"))]) => :end_time
	)

	test_net.duration = (test_net.end_time .- test_net.start_time) ./ 60000

	test_net[!, :task] .= "Post-PILT test net"


	# WM duration
	WM = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "WM_instructions"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "PILT"))]) => :end_time
	)

	WM.duration = (WM.end_time .- WM.start_time) ./ 60000

	WM[!, :task] .= "WM"

	# WM net duration
	WM_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "WM_instructions"))]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "PILT"))]) => :end_time
	)

	WM_net.duration = (WM_net.end_time .- WM_net.start_time) ./ 60000

	WM_net[!, :task] .= "WM net"


	# Reversal duration
	reversal = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findfirst(.!ismissing.(p) .&& (p .== "reversal_instruction"))-1]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "reversal"))]) => :end_time
	)

	reversal.duration = (reversal.end_time .- reversal.start_time) ./ 60000

	reversal[!, :task] .= "Reversal"

	# Reversal net duration
	reversal_net = combine(
		gb,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "reversal_instruction"))]) => 
				:start_time,
		[:trialphase, :time_elapsed] => 
			((p, t) -> t[findlast(.!ismissing.(p) .&& (p .== "reversal"))]) => :end_time
	)

	reversal_net.duration = (reversal_net.end_time .- reversal_net.start_time) ./ 60000

	reversal_net[!, :task] .= "Reversal net"


	# Combine together

	durations = vcat([PILT, PILT_net, vigour, vigour_net, PIT, PIT_net, test, test_net, WM, WM_net, reversal, reversal_net]...)
end

# ╔═╡ 0c617ddc-45f3-45f6-9c06-37c65ed6764e
let

	dur_sum = combine(
		groupby(durations, :task),
		:duration => median => :duration,
		:duration => lb => :lb,
		:duration => ub => :ub,
		:duration => llb => :llb,
		:duration => uub => :uub
	)

	tasks = filter(x -> !occursin("net", x), unique(dur_sum.task))

	fs = []
	for (i, t) in enumerate(tasks)

		pdat = filter(x -> (x.task in tasks) || x.task == "$(t) net", dur_sum)

		transform!(pdat, :task => ByRow(x -> x in [t, "$(t) net"]) => :highlight)

		categories = vcat(tasks[1:i], "$(t) net" => "—excl. instruct.", tasks[i+1:end])
		
		mp = data(pdat) * 
		(
			mapping(
				:task,
				:llb,
				:uub,
				color = :highlight
			) * visual(Rangebars, direction = :x) +
			mapping(
				:task,
				:lb,
				:ub,
				color = :highlight
			) * visual(Rangebars, direction = :x, linewidth = 3) +
			mapping(
				:duration,
				:task,
				color = :highlight
			) * visual(Scatter)
		)
		
		f = Figure(size = (18.78, 14.78) .* 36 ./ 2.54, figure_padding = 0)
		
		draw!(f[1, 1], mp, scales(
				Color = (; palette = [:grey, Makie.wong_colors()[1]]),
				Y = (; 
						categories = reverse(categories)
		)); axis = (; xlabel = "Duration (m)", ylabel = ""))

		# Save
		filepath = "results/workshop/duration_$t.png"
		save(filepath, f, pt_per_unit = 1)

		# upload_to_osf(
		# 	filepath,
		# 	proj,
		# 	osf_folder
		# )

	
		push!(fs, f)
	end

	fs

end

# ╔═╡ b978543e-42c7-4a98-bdf0-cfaf3bd83da3
md"""
## Export acceaptability and time elapsed data
"""

# ╔═╡ c5facfa0-8694-48b4-a4d8-c21645dcc22c
let
	accept_data = @chain p_sum begin
		@filter(!ismissing(finished) & finished)
		@select(prolific_pid, session, matches("_(difficulty|clear|enjoy)"))
	end
	dur_data = unstack(durations, [:prolific_pid, :session], :task, :duration, renamecols=x->Symbol(x, :_dur))
	CSV.write("results/workshop/acceptability.csv", accept_data)
	CSV.write("results/workshop/time_elapsed.csv", dur_data)
end

# ╔═╡ Cell order:
# ╠═fce1a1b4-a5cf-11ef-0068-c7282cd862f0
# ╠═0978c5a9-b488-44f0-8a6c-9d3e51da4c3a
# ╠═aae6bde5-84f8-4ecd-8ec9-bc1cf7f92583
# ╠═caa2e442-9eff-4c4d-a6f2-b09cfccf2357
# ╠═8191dbc5-a8dd-49b0-bd30-58c1d4744f74
# ╟─23f4c513-c4af-4bde-adac-8cdd88e48333
# ╠═b083eaa9-9b09-423c-bc32-9c5f17a91391
# ╠═3f94af37-bc0d-4bd9-ba4c-5fa244e2b3a3
# ╠═565e9389-f251-4df0-9e4d-8fab4e009611
# ╠═864f7f16-7408-441c-89bf-c0e6894ffce4
# ╟─d71d62c5-f617-4a5a-a27c-8a9820347b76
# ╠═e2711875-18a0-4c16-835a-9e12d8f6316e
# ╠═15950e90-11b7-45fb-9507-8f5e805d6ce4
# ╠═0c617ddc-45f3-45f6-9c06-37c65ed6764e
# ╠═2481764a-be2c-413b-bd48-e460c00fe2ff
# ╟─f51aa34a-5501-41f2-b12f-4340d0cdaf26
# ╟─b978543e-42c7-4a98-bdf0-cfaf3bd83da3
# ╠═c5facfa0-8694-48b4-a4d8-c21645dcc22c
