begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	using Tidier
	println(pwd())
	include(joinpath(pwd(), "fetch_preprocess_data.jl"))
	include(joinpath(pwd(), "sample_utils.jl"))
	include(joinpath(pwd(), "plotting_utils.jl"))
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

# Load WM and LTM data
begin
	_, _, _, _, _, _, _, _, jspsych_data = load_pilot8_data(; force_download = false, return_version = "0.2")
	nothing
end

# Compute LTM and WM durations
WM_LTM_durations = let
	timestamps = combine(
		groupby(jspsych_data, :prolific_pid),
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "LTM_instructions").(tp)) - 1]) => 
			:LTM_instructions_start,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "ltm").(tp)) - 1]) => 
			:LTM_start,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findlast((x -> !ismissing(x) && x == "ltm").(tp))]) => 
			:LTM_end,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> begin
				idx = findfirst((x -> !ismissing(x) && x == "WM_instructions").(tp))
				isnothing(idx) ? missing : t[idx - 1]
			end) => 
			:WM_instructions_start,
		[:trialphase, :time_elapsed] => 
				((tp, t) -> begin
				idx = findfirst((x -> !ismissing(x) && x == "wm").(tp))
				isnothing(idx) ? missing : t[idx - 1]
			end) => 
			:WM_start,
		[:trialphase, :time_elapsed] => 
				((tp, t) -> begin
				idx = findlast((x -> !ismissing(x) && x == "wm").(tp))
				isnothing(idx) ? missing : t[idx]
			end) => 
			:WM_end,
	)

	# Calculate durations
	timestamps.LTM_instructions = (timestamps.LTM_start .- timestamps.LTM_instructions_start) ./ 1000 ./ 60
	timestamps.WM_instructions = (timestamps.WM_start .- timestamps.WM_instructions_start) ./ 1000 ./ 60
	timestamps.LTM = (timestamps.LTM_end .- timestamps.LTM_start) ./ 1000 ./ 60
	timestamps.WM = (timestamps.WM_end .- timestamps.WM_start) ./ 1000 ./ 60
	
	# Wide to long
	durations = stack(
		timestamps, 
		[:LTM_instructions, :WM_instructions, :LTM, :WM], 
		[:prolific_pid],
		value_name = :duration
	)

	# Part variable: whether instructions or task
	durations.part = ifelse.(
		occursin.(r"instructions", durations.variable),
		"Instructions",
		"Task"
	)

	# Task variable: whether LTM or WM
	durations.task = ifelse.(
		occursin.(r"LTM", durations.variable),
		"LTM",
		"WM"
	)

	# Remove missing values
	filter!(x -> !ismissing(x.duration), durations)

	# Add SD
	DataFrames.transform!(
		groupby(durations, [:task, :part]),
		:duration => std => :sd
	)
	
end

# Plot duration rainclouds
let df = copy(WM_LTM_durations)
	f = Figure()

	# Remove extremes in instructions
	filter!(x -> x.part != "Instructions" || (x.duration < 20), df)

	df.side = ifelse.(df.part .== "Instructions", :right, :left)

	mp = data(df) * mapping(
		:task => "Task",
		:duration => "Duration (minutes)",
		color = :part => ""
		) * visual(RainClouds, plot_boxplots = false)

	plt = draw!(f[1,1], mp)

	legend!(f[0,1], plt, tellwidth = false, halign = 0.5, orientation = :horizontal, framevisible = false, titleposition = :left)
	
	f
end