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

# Load Pilot 7
begin
	_, _, _, _, _, _, _, pilot7_jspsych_data = load_pilot7_data(return_version = "7.0")
	pilot7_jspsych_data = exclude_double_takers(pilot7_jspsych_data)
	sort!(pilot7_jspsych_data, [:prolific_pid, :exp_start_time, :trial_index])
end

# Load Pilot 8
begin
	_, _, _, _, _, _, _, _, pilot8_jspsych_data = load_pilot8_data(; force_download = false, return_version = "0.2")
	pilot8_jspsych_data = exclude_double_takers(pilot8_jspsych_data)
	nothing
end

# Load control pilot 2
begin
	_, _, control2_1 = load_control_pilot2_data()
	control2_1 = exclude_double_takers(control2_1)
	_, _, control2_3 = load_control_pilot2_data(; session = "3")
	control2_3 = exclude_double_takers(control2_3)

	# Make sure we don't have the same prolific_pid in both versions
	@assert length(intersect(control2_1.prolific_pid, control2_3.prolific_pid)) == 0 "Control 2 sessions have overlapping participants"

	# Merge data
	control2 = vcat(control2_1, control2_3)
end

function calculate_durations(
	task::String;
	df::AbstractDataFrame,
	trial_counting_finder::Pair{Vector{Symbol}, <:Function} = [:trialphase] => (x -> sum((.!ismissing.(x)) .&& (x .== lowercase(task)))),
	n_trials_criterion::Int,
	instruction_start_finder::Pair{Vector{Symbol}, <:Function},
	task_start_finder::Pair{Vector{Symbol}, <:Function},
	task_end_finder::Pair{Vector{Symbol}, <:Function},
	extreme_value_criterion::Int = 30
)
	# Exclude non finishers
	participants = combine(
		groupby(df, [:prolific_pid, :session]),
		trial_counting_finder[1] => trial_counting_finder[2] => :n_trials
	)

	filter!(x -> x.n_trials == n_trials_criterion, participants)

	@assert nrow(participants) > 0 "No participants with $task data found"

	# Filter to keep only rows with pid-session combinations in reversal_participants
	task_data = semijoin(
		df, 
		participants, 
		on = [:prolific_pid, :session]
	)

	timestamps = combine(
		groupby(task_data, [:prolific_pid, :session]),
		instruction_start_finder[1] => instruction_start_finder[2] => :instructions_start,
		task_start_finder[1] => task_start_finder[2] => :task_start,
		task_end_finder[1] => task_end_finder[2] =>  :task_end,
	)

	# Calculate durations
	timestamps[!, Symbol("$(task)_instructions")] = (timestamps.task_start .- timestamps.instructions_start) ./ 1000 ./ 60
	timestamps[!, Symbol("$task")] = (timestamps.task_end .- timestamps.task_start) ./ 1000 ./ 60
	
	# Wide to long
	durations = stack(
		timestamps, 
		[Symbol("$(task)_instructions"), Symbol("$task")], 
		[:prolific_pid, :session],
		value_name = :duration
	)

	@assert minimum(durations.duration) > 0 "Negative duration found for $task"

	# Part variable: whether instructions or task
	durations.part = ifelse.(
		occursin.(r"instructions", durations.variable),
		"Instructions",
		"Task"
	)

	# Task variable
	if length(unique(durations.session)) > 1
		durations[!, :task] = ifelse.(
			durations.session .== "1",
			"$task sess. 1",
			"$task sess. 2"
		)
	else
		durations[!, :task] .= "$task"
	end

	# Remove missing values
	@info "Removing $(sum(ismissing(durations.duration))) missing values"
	filter!(x -> !ismissing(x.duration), durations)

	# Remove extremes
	filter!(x -> x.duration < extreme_value_criterion, durations)
	
end

wm_durations = calculate_durations(
	"WM";
	df = pilot8_jspsych_data,
	n_trials_criterion = 108,
	instruction_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> begin
			idx = findfirst((x -> !ismissing(x) && x == "WM_instructions").(tp))
			isnothing(idx) ? missing : t[idx - 1]
		end),
	task_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> begin
		idx = findfirst((x -> !ismissing(x) && x == "wm").(tp))
		isnothing(idx) ? missing : t[idx - 1]
	end),
	task_end_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> begin
		idx = findlast((x -> !ismissing(x) && x == "wm").(tp))
		isnothing(idx) ? missing : t[idx]
	end)
)

ltm_durations = calculate_durations(
	"LTM";
	df = pilot8_jspsych_data,
	n_trials_criterion = 108,
	instruction_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "LTM_instructions").(tp)) - 1]),
	task_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "ltm").(tp)) - 1]),
	task_end_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> t[findlast((x -> !ismissing(x) && x == "ltm").(tp))])
)


pilt_early_stop_durations = calculate_durations(
	"PILT early stop";
	trial_counting_finder = [:trialphase, :n_stimuli, :block] => (tp, ns, bl) -> maximum(bl[.!ismissing.(tp) .&& (tp .== "PILT") .&& .!ismissing.(ns) .&& (ns .== 2) .&& isa.(bl, Number)]),
	df = pilot6_jspsych_data,
	n_trials_criterion = 20, # Actually blocks
	instruction_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "instruction").(tp)) - 1]),
	task_start_finder = [:block, :time_elapsed] => 
		((b, t) -> t[findfirst((x -> !ismissing(x) && x == 1).(b)) - 1]),
	task_end_finder = [:trialphase, :n_stimuli, :block, :time_elapsed] => 
		((tp, ns, b, t) -> begin
			idx = findlast((i -> !ismissing(tp[i]) && tp[i] == "PILT" && 
							!ismissing(ns[i]) && ns[i] == 2 && 
							!ismissing(b[i]) && b[i] == 20), 1:length(tp))
			isnothing(idx) ? missing : t[idx]
		end)
)

pilt_no_early_stop_durations = calculate_durations(
	"PILT no early stop";
	trial_counting_finder = [:trialphase, :n_stimuli, :block] => (tp, ns, bl) -> maximum(bl[.!ismissing.(tp) .&& (tp .== "PILT") .&& .!ismissing.(ns) .&& (ns .== 2) .&& isa.(bl, Number)]),
	df = pilot7_jspsych_data,
	n_trials_criterion = 20, # Actually blocks
	instruction_start_finder = [:trialphase, :time_elapsed] => 
		((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "instruction").(tp)) - 1]),
	task_start_finder = [:block, :time_elapsed] => 
		((b, t) -> t[findfirst((x -> !ismissing(x) && x == 1).(b)) - 1]),
	task_end_finder = [:trialphase, :n_stimuli, :block, :time_elapsed] => 
		((tp, ns, b, t) -> begin
			idx = findlast((i -> !ismissing(tp[i]) && tp[i] == "PILT" && 
							!ismissing(ns[i]) && ns[i] == 2 && 
							!ismissing(b[i]) && b[i] == 20), 1:length(tp))
			isnothing(idx) ? missing : t[idx]
		end)
)

reversal_durations = calculate_durations(
	"Reversal";
	df = pilot6_jspsych_data,
	n_trials_criterion = 150,
	instruction_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "reversal_instruction").(tp)) - 1]),
	task_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "reversal").(tp)) - 1]),
	task_end_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findlast((x -> !ismissing(x) && x == "reversal").(tp))])
)

pit_durations = calculate_durations(
	"PIT";
	df = pilot7_jspsych_data,
	trial_counting_finder = [:trialphase] =>  x -> sum((.!ismissing.(x)) .&& (x .== "pit_trial")),
	n_trials_criterion = 72,
	instruction_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "pit_instructions").(tp)) - 1]),
	task_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "pit_trial").(tp)) - 1]),
	task_end_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findlast((x -> !ismissing(x) && x == "pit_trial").(tp))])
)

vigour_durations = calculate_durations(
	"vigour";
	df = pilot7_jspsych_data,
	trial_counting_finder = [:trialphase] => x -> sum((.!ismissing.(x)) .&& (x .== "vigour_trial")),
	n_trials_criterion = 54,
	instruction_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "vigour_instructions").(tp)) - 1]),
	task_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "vigour_trial").(tp)) - 1]),
	task_end_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findlast((x -> !ismissing(x) && x == "vigour_trial").(tp))])
)

control_durations = calculate_durations(
	"Control";
	df = control2,
	trial_counting_finder = [:trialphase] =>  x -> sum((.!ismissing.(x)) .&& (x .== "control_bonus")),
	n_trials_criterion = 1,
	instruction_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "control_instructions").(tp)) - 1]),
	task_start_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "control_explore").(tp)) - 1]),
	task_end_finder = [:trialphase, :time_elapsed] => ((tp, t) -> t[findlast((x -> !ismissing(x) && x == "control_bonus").(tp)) - 1])
)


## Plot and summarize -----------------------------------|
# Combine durations from different experiments
begin
	durations = vcat(wm_durations, ltm_durations, vigour_durations, pit_durations, reversal_durations, pilt_early_stop_durations, pilt_no_early_stop_durations)
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