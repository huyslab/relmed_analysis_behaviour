# Setup
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Random, DataFrames, JSON, CSV, StatsBase, JLD2, HTTP, CairoMakie, Printf, Distributions, CategoricalArrays, AlgebraOfGraphics, Dates
	using LogExpFunctions: logistic, logit
	include("fetch_preprocess_data.jl")
	include("sample_utils.jl")
	include("questionnaire_utils.jl")
	nothing
end

# Download data from REDCap
begin
	PILT_data, WM_data, LTM_data, reversal_data, jspsych_data = load_trial1_data(; force_download = false)
	nothing
end

# Data dictionary for jsPsych data output
begin

	# Load variable descriptions
	variable_descriptions = CSV.read("task_validation/variable_descriptions.csv", DataFrame)

	# Convert variable names to symbols
	variable_descriptions.variable = Symbol.(variable_descriptions.variable)

	# Describe for data dictionary
	jspsych_data_dict = describe(filter(x -> x.record_id == "debugsimulate_2025-03-26_12:07:39", jspsych_data), :eltype, :nmissing, :min, :max)

	# Add variable descriptions
	leftjoin!(jspsych_data_dict, variable_descriptions, on = :variable)

	# Move variable descriptions to the front
	jspsych_data_dict = jspsych_data_dict[:, [:variable, :description, :eltype, :nmissing, :min, :max]]

	# Remove prolific_pid
	filter!(x -> x.variable != :prolific_pid, jspsych_data_dict)

	# Write the transformed data dictionary to a CSV file
	CSV.write(
		"results/task_validation/jspsych_data_dict.csv", 
		jspsych_data_dict,
		transform = (col, val) -> something(val, missing)
	)
end

begin
	# Separate questionnaire data
	questionnaire_data = extract_raw_questionnaire_data(jspsych_data)

	# Rename prolific_pid to participant_id
	rename!(questionnaire_data,
		:prolific_pid => :participant_id
	)

	CSV.write(
		"results/task_validation/example_questionnaire_data.csv", 
		questionnaire_data
	)
end

