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

## Load data --------------------------------------------|
# Load Pilot 6
begin
	# Load data
	_, _, _, _, _, _, _, pilot6_jspsych_data = load_pilot6_data()
	pilot6_jspsych_data = exclude_double_takers(pilot6_jspsych_data)
	nothing
end

# Load Pilot 8
begin
	_, _, _, _, _, _, _, _, pilot8_jspsych_data = load_pilot8_data(; force_download = false, return_version = "0.2")
	nothing
end

## WM and LTM data --------------------------------------------|
# Compute LTM and WM durations
WM_LTM_durations = let
	timestamps = combine(
		groupby(pilot8_jspsych_data, :prolific_pid),
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

	# Remove extremes in instructions
	filter!(x -> x.part != "Instructions" || (x.duration < 20), durations)

end

## PILT data --------------------------------------------|
PILT_durations = let
	timestamps = combine(
		groupby(pilot6_jspsych_data, [:prolific_pid, :session]),
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "instruction").(tp)) - 1]) => 
			:PILT_instructions_start,
		[:block, :time_elapsed] => 
			((b, t) -> t[findfirst((x -> !ismissing(x) && x == 1).(b)) - 1]) => 
			:PILT_start,
		[:trialphase, :n_stimuli, :block, :time_elapsed] => 
			((tp, ns, b, t) -> begin
				idx = findlast((i -> !ismissing(tp[i]) && tp[i] == "PILT" && 
								!ismissing(ns[i]) && ns[i] == 2 && 
								!ismissing(b[i]) && b[i] == 20), 1:length(tp))
				isnothing(idx) ? missing : t[idx]
			end) => 
			:PILT_end,
	)

	# Calculate durations
	timestamps.PILT_instructions = (timestamps.PILT_start .- timestamps.PILT_instructions_start) ./ 1000 ./ 60
	timestamps.PILT = (timestamps.PILT_end .- timestamps.PILT_start) ./ 1000 ./ 60
	
	# Wide to long
	durations = stack(
		timestamps, 
		[:PILT_instructions, :PILT], 
		[:prolific_pid, :session],
		value_name = :duration
	)

	# Part variable: whether instructions or task
	durations.part = ifelse.(
		occursin.(r"instructions", durations.variable),
		"Instructions",
		"Task"
	)

	# Task variable: whether LTM or WM
	durations[!, :task] = ifelse.(
		durations.session .== "1",
		"PILT\nsess. 1",
		"PILT\nsess. 2"
	)

	# Remove missing values
	filter!(x -> !ismissing(x.duration), durations)

	# Remove extremes
	filter!(x -> x.duration < 30, durations)
	
end

# Reversal data --------------------------------------------|
reversal_durations = let

	# Exclude non finishers
	reversal_participants = combine(
		groupby(pilot6_jspsych_data, [:prolific_pid, :session]),
		:trialphase => (x -> sum((.!ismissing.(x)) .&& (x .== "reversal"))) => :n_trials
	)

	filter!(x -> x.n_trials == 150, reversal_participants)

	# Filter to keep only rows with pid-session combinations in reversal_participants
	reversal_data = semijoin(
		pilot6_jspsych_data, 
		reversal_participants, 
		on = [:prolific_pid, :session]
	)

	timestamps = combine(
		groupby(reversal_data, [:prolific_pid, :session]),
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "reversal_instruction").(tp)) - 1]) => 
			:reversal_instructions_start,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findlast((x -> !ismissing(x) && x == "reversal_instruction").(tp))]) => 
			:reversal_start,
		[:trialphase, :time_elapsed] => 
		((tp, t) -> t[findlast((x -> !ismissing(x) && x == "reversal").(tp))]) =>  
			:reversal_end,
	)

	# Calculate durations
	timestamps.reversal_instructions = (timestamps.reversal_start .- timestamps.reversal_instructions_start) ./ 1000 ./ 60
	timestamps.reversal = (timestamps.reversal_end .- timestamps.reversal_start) ./ 1000 ./ 60
	
	# Wide to long
	durations = stack(
		timestamps, 
		[:reversal_instructions, :reversal], 
		[:prolific_pid, :session],
		value_name = :duration
	)

	# Part variable: whether instructions or task
	durations.part = ifelse.(
		occursin.(r"instructions", durations.variable),
		"Instructions",
		"Task"
	)

	# Task variable: whether LTM or WM
	durations[!, :task] = ifelse.(
		durations.session .== "1",
		"Reversal\nsess. 1",
		"Reversal\nsess. 2"
	)

	# Remove missing values
	filter!(x -> !ismissing(x.duration), durations)

	# Remove extremes
	filter!(x -> x.duration < 30, durations)
	
end


## Plot and summarize -----------------------------------|
# Combine durations from different experiments
begin
	durations = vcat(WM_LTM_durations, select(PILT_durations, Not(:session)), select(reversal_durations, Not(:session)))
end

# Plot duration rainclouds
let df = copy(durations)
	f = Figure()

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

# Percentile table of durations
let df = copy(durations)
	# Calculate percentiles
	pctiles = combine(
		groupby(df, [:task, :part]),
		:duration => (x -> quantile(x, 0.5)) => :p50,
		:duration => (x -> quantile(x, 0.75)) => :p75,
		:duration => (x -> quantile(x, 0.9)) => :p90,
		:duration => (x -> quantile(x, 0.95)) => :p95
	)

	# Format values as minutes and seconds
	pctiles.p50 = (x -> "$(floor(x))'$(Int(round((x - floor(x)) * 60)))\"").(pctiles.p50)
	pctiles.p75 = (x -> "$(floor(x))'$(Int(round((x - floor(x)) * 60)))\"").(pctiles.p75)
	pctiles.p90 = (x -> "$(floor(x))'$(Int(round((x - floor(x)) * 60)))\"").(pctiles.p90)
	pctiles.p95 = (x -> "$(floor(x))'$(Int(round((x - floor(x)) * 60)))\"").(pctiles.p95)

	println(pctiles)
	
end