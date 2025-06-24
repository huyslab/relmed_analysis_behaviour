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