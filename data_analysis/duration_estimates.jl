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
			:ltm_instructions_start,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findfirst((x -> !ismissing(x) && x == "ltm").(tp)) - 1]) => 
			:ltm_start,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> t[findlast((x -> !ismissing(x) && x == "ltm").(tp))]) => 
			:ltm_end,
		[:trialphase, :time_elapsed] => 
			((tp, t) -> begin
				idx = findfirst((x -> !ismissing(x) && x == "WM_instructions").(tp))
				isnothing(idx) ? missing : t[idx - 1]
			end) => 
			:wm_instructions_start,
		[:trialphase, :time_elapsed] => 
				((tp, t) -> begin
				idx = findfirst((x -> !ismissing(x) && x == "wm").(tp))
				isnothing(idx) ? missing : t[idx - 1]
			end) => 
			:wm_start,
		[:trialphase, :time_elapsed] => 
				((tp, t) -> begin
				idx = findfirst((x -> !ismissing(x) && x == "wm").(tp))
				isnothing(idx) ? missing : t[idx]
			end) => 
			:wm_end,
	)

	timestamps.ltm_instructions = (timestamps.ltm_start .- timestamps.ltm_instructions_start) ./ 1000 ./ 60
	timestamps.wm_instructions = (timestamps.wm_start .- timestamps.wm_instructions_start) ./ 1000 ./ 60
	timestamps.ltm_duration = (timestamps.ltm_end .- timestamps.ltm_start) ./ 1000 ./ 60
	timestamps.wm_duration = (timestamps.wm_end .- timestamps.wm_start) ./ 1000 ./ 60

	select(
		timestamps,
		:prolific_pid,
		:ltm_instructions,
		:ltm_duration,
		:wm_instructions,
		:wm_duration,
	)
end
