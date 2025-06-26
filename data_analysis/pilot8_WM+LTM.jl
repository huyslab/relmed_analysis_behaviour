begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	nothing
end

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

# Helper functions
begin
	recoder = (x, edges, labels) -> ([findfirst(v ≤ edge for edge in edges) === nothing ? labels[end] : labels[findfirst(v ≤ edge for edge in edges)] for v in x])

	function compute_delays(vec::AbstractVector)
		last_seen = Dict{Any, Int}()
		delays = zeros(Int, length(vec))

		for (i, val) in enumerate(vec)
			delays[i] = haskey(last_seen, val) ? (i - last_seen[val]) : 0
			last_seen[val] = i
		end

		return delays
	end

	function clean_WM_LTM_data(
		df::AbstractDataFrame
	)
		# Clean data
		data_clean = exclude_PLT_sessions(df, required_n_blocks = 1)

		# Sort
		sort!(
			data_clean,
			[:prolific_pid, :session, :block, :trial]
		)

		# Apperance number
		transform!(
			groupby(data_clean, [:prolific_pid, :exp_start_time, :session, :block, :stimulus_group]),
			:trial => (x -> 1:length(x)) => :appearance
		)

		# Compute delays
		DataFrames.transform!(
			groupby(
				data_clean,
				:prolific_pid
			),
			:stimulus_group => compute_delays => :delay,
		) 

		data_clean = filter(x -> x.response != "noresp", data_clean)

		# Previous correct
		DataFrames.transform!(
			groupby(
				data_clean,
				[:prolific_pid, :stimulus_group]
			),
			:response_optimal => lag => :previous_optimal,
		)

	end
end

# Load data
begin
	_, WM_data, LTM_data, WM_test_data, LTM_test_data, _, control_task_data, _, _ = load_pilot8_data(; force_download = false, return_version = "0.2")
	nothing
end


# Clean and prepare data, and combine
data_clean  = let
	WM_data_clean, LTM_data_clean = clean_WM_LTM_data.([WM_data, LTM_data])

	# Indicator variable
	WM_data_clean.task .= "1 stim"

	LTM_data_clean.task .= "3 stims"

	data_clean = vcat(
		WM_data_clean,
		LTM_data_clean
	)
end

# Plot learning curve
let df = data_clean

	# Create figure
	f = Figure()

	
	# Summarize by appearance
	app_curve = combine(
		groupby(df, [:prolific_pid, :task, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize by apperance and n_groups
	app_curve_sum = combine(
		groupby(app_curve, [:task, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:task, :appearance])

	# Create mapping
	mp1 = (data(app_curve_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :task => "Task"
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:acc,
			color = :task => "Task"
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1; axis=(; ylabel = "Prop. optimal choice ±SE"))

	legend!(f[0,1], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)


	f
end

# Plot learning curve with delay bins
let df = data_clean
	
	df.delay_bin = recoder(df.delay, [0, 1, 5, 10], ["0", "1", "2-5", "6-10"])
	
	# Summarize by participant
	app_curve = combine(
		groupby(df, [:prolific_pid, :task, :delay_bin, :appearance]),
		:response_optimal => mean => :acc
	)

	# Summarize across participants
	app_curve_sum = combine(
		groupby(app_curve, [:task, :delay_bin, :appearance]),
		:acc => mean => :acc,
		:acc => sem => :se
	)

	# Compute bounds
	app_curve_sum.lb = app_curve_sum.acc .- app_curve_sum.se
	app_curve_sum.ub = app_curve_sum.acc .+ app_curve_sum.se

	# Sort
	sort!(app_curve_sum, [:task, :delay_bin, :appearance])

	# Create mapping
	mp2 = (data(app_curve_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:acc => "Prop. optimal choice",
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Lines))) + (
		data(filter(x -> x.delay_bin == "0", app_curve_sum)) *
		(mapping(
			:appearance  => "Apperance #",
			:acc,
			:se,
			color = :delay_bin => "Delay",
			col = :task
		) * visual(Errorbars) +
		mapping(
			:appearance  => "Apperance #",
			:acc,
			color = :delay_bin  => "Delay",
			col = :task
		) * visual(Scatter))
	)

	f = Figure()

	plt = draw!(f[1,1], mp2; axis=(; ylabel = "Prop. optimal choice ±SE"))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end

# Plot RT by apperance
let df = copy(data_clean)

	# Create figure
	f = Figure()

	# Nice labels for response_optimal
	df.response = ifelse.(
		df.response_optimal,
		"Correct",
		"Error"
	)

	# Summarize by appearance
	rt_app = combine(
		groupby(
			filter(x -> x.rt > 200, df), 
			[:prolific_pid, :task, :appearance, :response]
		),
		:rt => mean => :rt
	)

	# Summarize by apperance and n_groups
	rt_app_sum = combine(
		groupby(rt_app, [:task, :appearance, :response]),
		:rt => mean => :rt,
		:rt => sem => :se
	)

	# Compute bounds
	rt_app_sum.lb = rt_app_sum.rt .- rt_app_sum.se
	rt_app_sum.ub = rt_app_sum.rt .+ rt_app_sum.se

	# Sort
	sort!(rt_app_sum, [:task, :response, :appearance])

	# Create mapping
	mp1 = (data(rt_app_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :task => "Task",
			col = :response
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:rt,
			color = :task => "Task",
			col = :response
	) * visual(Lines)))
	
	# Plot
	plt1 = draw!(f[1,1], mp1; axis=(; ylabel = "RT (mean±SE)"))

	legend!(f[0,1], plt1, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)


	f
end

# Plot RT by appearance and delay bins
let df = copy(data_clean)
	
	# Bin delays
	df.delay_bin = recoder(df.delay, [0, 1, 5, 10], ["0", "1", "2-5", "6-10"])
	
	# Summarize by participant
	rt_app_delay = combine(
		groupby(
			filter(x -> x.response_optimal, df), 
			[:prolific_pid, :task, :delay_bin, :appearance]
		),
		:rt => mean => :rt
	)

	# Summarize across participants
	rt_app_delay_sum = combine(
		groupby(rt_app_delay, [:task, :delay_bin, :appearance]),
		:rt => mean => :rt,
		:rt => sem => :se
	)

	# Compute bounds
	rt_app_delay_sum.lb = rt_app_delay_sum.rt .- rt_app_delay_sum.se
	rt_app_delay_sum.ub = rt_app_delay_sum.rt .+ rt_app_delay_sum.se

	# Sort
	sort!(rt_app_delay_sum, [:task, :delay_bin, :appearance])

	# Create mapping
	mp2 = (data(rt_app_delay_sum) * (
		mapping(
			:appearance => "Apperance #",
			:lb,
			:ub,
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Band, alpha = 0.5) +
		mapping(
			:appearance => "Apperance #",
			:rt,
			color = :delay_bin  => "Delay",
			col = :task
	) * visual(Lines))) + (
		data(filter(x -> x.delay_bin == "0", rt_app_delay_sum)) *
		(mapping(
			:appearance  => "Apperance #",
			:rt,
			:se,
			color = :delay_bin => "Delay",
			col = :task
		) * visual(Errorbars) +
		mapping(
			:appearance  => "Apperance #",
			:rt,
			color = :delay_bin  => "Delay",
			col = :task
		) * visual(Scatter))
	)

	f = Figure()

	plt = draw!(f[1,1], mp2; axis=(; ylabel = "RT (mean±SE)"))

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)

	f
end

function summarize_value_merge_to_test(
	df::AbstractDataFrame,
	test_df::AbstractDataFrame;
)

	# Summarize value by participant and stimulus
	value_sum = combine(
		groupby(df, [:prolific_pid, :chosen_stimulus]),
		:chosen_feedback => mean => :value,
		:chosen_feedback => StatsBase.mode => :common_outcome
	)

	# Merge with test data
	for side in ["left", "right"]
		
		leftjoin!(
			test_df,
			rename(
				value_sum, 
				:chosen_stimulus => Symbol("stimulus_$side"),
				:value => Symbol("value_$side"),
				:common_outcome => Symbol("common_outcome_$side")
			),
			on = [:prolific_pid, Symbol("stimulus_$side")]
		)
	end

	# Compute Δ value
	test_df.Δ_value = test_df.value_right .- test_df.value_left


	# Summarize apperance by participants and stimulus group
	apperance_sum = combine(
		groupby(df, [:prolific_pid, :stimulus_group_id]),
		:chosen_feedback => length => :n_trials,
	)

	stimulus_group = unique(vcat(
		[
			unique(select(
				df, 
				:stimulus_group_id,
				Symbol("stimulus_$side") => :stimulus
			))
			for side in ["left", "middle", "right"]
		]...
	))

	# Merge with apperance sum
	apperance_sum = innerjoin(
		apperance_sum,
		stimulus_group,
		on = :stimulus_group_id
	)

	# Merge with test data
	for side in ["left", "right"]
		leftjoin!(
			test_df,
			select(
				apperance_sum, 
				:prolific_pid,
				:stimulus => Symbol("stimulus_$side"),
				:n_trials => Symbol("n_trials_$side")
			),
			on = [:prolific_pid, Symbol("stimulus_$side")]
		)
	end
	
	
	# Compute right chosen
	test_df.right_chosen = test_df.response .== "right"

	dropmissing!(test_df, [:value_left, :value_right, :Δ_value, :right_chosen])

	return test_df
end


# Prepare test data
test_df = let 
	# Summarize value and merge with test data
	WM_test = summarize_value_merge_to_test(
		filter(x -> x.task == "1 stim", data_clean),
		copy(WM_test_data)
	)

	LTM_test = summarize_value_merge_to_test(
		filter(x -> x.task == "3 stims", data_clean),
		copy(LTM_test_data)
	)

	WM_test.task .= "1 stim"
	LTM_test.task .= "3 stims"

	# Combine test data
	test_df = vcat(
		WM_test,
		LTM_test
	)

	filter!(x -> x.response != "noresp", test_df)
end

function quantile_bin_mean(var::AbstractVector; n_bins::Int)

	found_bins = false
	tn_bins = n_bins
	edges = Float64[]
	while !found_bins
		edges = unique(quantile(var, range(0, 1; length=tn_bins+1)))

		if length(edges) >= (n_bins + 1)
			found_bins = true
		else
			tn_bins += 1
		end
	end
	bin = cut(var, edges; extend=true)
	bin_mean = [mean(var[bin .== b]) for b in bin]
	return bin_mean
end


# Plot test data by Δ value
let n_bins = 8, test_df = copy(test_df)

	# Bin into equiprobable bins
	DataFrames.transform!(
		groupby(test_df, :task),
		:Δ_value => (x -> quantile_bin_mean(x; n_bins = n_bins)) => :Δ_value_bin
	)

	# Summarize by bin
	test_val = combine(
		groupby(test_df, [:prolific_pid, :Δ_value_bin, :task]),
		:right_chosen => mean => :right_chosen
	)

	test_val_sum = combine(
		groupby(test_val, [:Δ_value_bin, :task]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Plot
	mp = (data(test_val_sum) * (
		mapping(
			:Δ_value_bin,
			:right_chosen,
			:se,
			color = :task => "Task"
	) * visual(Errorbars) +
		mapping(
			:Δ_value_bin,
			:right_chosen,
			color = :task => "Task"
	) * visual(Scatter)))

	f = Figure()
	plt = draw!(f[1,1], mp; axis=(; xlabel = "Δ stimulus value\nright - left", ylabel = "Prop. right chosen ±SE"))
	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
	f
end

let n_bins = 4, test_df = copy(test_df)

	# Absolute Δ value
	test_df.abs_Δ_value = abs.(test_df.Δ_value)

	# Bin into equiprobable bins
	DataFrames.transform!(
		groupby(test_df, :task),
		[:abs_Δ_value, :task] => ((x,t) -> quantile_bin_mean(x; n_bins = only(unique(t)) == "1 stim" ? 4 : 8)) => :Δ_value_bin
	)

	# Accuracy variable
	test_df.accuracy = ifelse.(
		test_df.Δ_value .== 0,
		true,
		ifelse.(
			test_df.Δ_value .> 0,
			test_df.right_chosen,
			.!test_df.right_chosen
		)
	)

	# Summarize by bin
	test_val = combine(
		groupby(test_df, [:prolific_pid, :Δ_value_bin, :accuracy, :task]),
		:rt => mean => :rt
	)

	test_val_sum = combine(
		groupby(test_val, [:Δ_value_bin, :accuracy, :task]),
		:rt => mean => :rt,
		:rt => sem => :se
	)

	# Look only at accurate choices
	filter!(x -> x.accuracy, test_val_sum)

	# Plot
	mp = (data(test_val_sum) * (
		mapping(
			:Δ_value_bin,
			:rt,
			:se,
			color = :task => "Task"
	) * visual(Errorbars) +
		mapping(
			:Δ_value_bin,
			:rt,
			color = :task => "Task"
	) * visual(Scatter)))

	f = Figure()
	plt = draw!(f[1,1], mp; axis=(; xlabel = "|Δ stimulus value|", ylabel = "RT (mean±SE)"))
	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
	f
end

# Plot test data by # of appearances
let n_bins = 5,
	test_df = copy(test_df)

	# Compute apperance difference
	test_df.Δ_appearance = test_df.n_trials_right .- test_df.n_trials_left

	# Bin into equiprobable bins
	DataFrames.transform!(
		groupby(test_df, :task),
		:Δ_appearance => (x -> quantile_bin_mean(x; n_bins = n_bins)) => :Δ_appearance
	)

	# Summarize by bin
	test_val = combine(
		groupby(test_df, [:prolific_pid, :Δ_appearance, :task]),
		:right_chosen => mean => :right_chosen
	)

	test_val_sum = combine(
		groupby(test_val, [:Δ_appearance, :task]),
		:right_chosen => mean => :right_chosen,
		:right_chosen => sem => :se
	)

	# Plot
	mp = (data(test_val_sum) * (
		mapping(
			:Δ_appearance,
			:right_chosen,
			:se,
			color = :task => "Task"
	) * visual(Errorbars) +
		mapping(
			:Δ_appearance,
			:right_chosen,
			color = :task => "Task"
	) * visual(Scatter)))

	f = Figure()
	plt = draw!(f[1,1], mp; axis=(; xlabel = "Δ learning stage appearances\nright - left", ylabel = "Prop. right chosen ±SE"))
	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
	f
end

# Plot test data by Δ value
let test_df = copy(test_df)

	test_df.outcome_pair = string.(sort.([[r.common_outcome_left, r.common_outcome_right] for r in eachrow(test_df)]))

	# Higher chosen variable
	test_df.higher_chosen = ifelse.(
		test_df.common_outcome_right .≈ test_df.common_outcome_left,
		0.5,
		ifelse.(
			test_df.common_outcome_right .> test_df.common_outcome_left,
			test_df.right_chosen,
			.!test_df.right_chosen
		)
	)
	
	# Summarize by outcome
	test_val = combine(
		groupby(test_df, [:prolific_pid, :outcome_pair, :task]),
		:higher_chosen => mean => :higher_chosen
	)


	test_val_sum = combine(
		groupby(test_val, [:outcome_pair, :task]),
		:higher_chosen => mean => :higher_chosen,
		:higher_chosen => sem => :se
	)

	# Plot
	mp = (data(test_val_sum) * (
		mapping(
			:outcome_pair,
			:higher_chosen,
			:se,
			color = :task => "Task"
	) * visual(Errorbars) +
		mapping(
			:outcome_pair,
			:higher_chosen,
			color = :task => "Task"
	) * visual(Scatter)))

	f = Figure()
	plt = draw!(f[1,1], mp; axis=(; xlabel = "Common outcome for each choice", ylabel = "Prop. higher chosen ±SE"))
	
	
	# Plot RT
	# Summarize by outcome
	rt_out = combine(
		groupby(test_df, [:prolific_pid, :outcome_pair, :higher_chosen, :task]),
		:rt => mean => :rt
	)


	rt_out_sum = combine(
		groupby(rt_out, [:outcome_pair, :higher_chosen, :task]),
		:rt => mean => :rt,
		:rt => sem => :se
	)

	filter!(x -> x.higher_chosen > 0, rt_out_sum)

	mp = (data(rt_out_sum) * (
		mapping(
			:outcome_pair,
			:rt,
			:se,
			color = :task => "Task"
	) * visual(Errorbars) +
		mapping(
			:outcome_pair,
			:rt,
			color = :task => "Task"
	) * visual(Scatter)))
	
	plt = draw!(f[2,1], mp; axis=(; xlabel = "Common outcome for each choice", ylabel = "RT (mean±SE)"))

	
	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
	f
end
