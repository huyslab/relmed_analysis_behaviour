# Functions for fetching data and preprocessing it

# Fetch one data file by record_id
function get_REDCap_file(
	record_id::String
)
	# Create the payload for getting the file
	file_payload = Dict(
		"token" => ENV["REDCap_token"],
	    "content" => "file",
		"action" => "export",
		"record" => record_id,
		"field" => "other_data",
		"returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	file = HTTP.post(ENV["REDCap_url"], body=HTTP.Form(file_payload), verbose = true)

	# Parse
	return JSON.parse(String(file.body))
end

# Fetch entire dataset
function get_REDCap_data()

	# Get the records --------
	# Create the payload for getting the record details
	rec_payload = Dict(
		"token" => ENV["REDCap_token"],
	    "content" => "record",
	    "action" => "export",
	    "format" => "json",
	    "type" => "flat",
	    "csvDelimiter" => "",
	    "rawOrLabel" => "raw",
	    "rawOrLabelHeaders" => "raw",
	    "exportCheckboxLabel" => "false",
	    "exportSurveyFields" => "false",
	    "exportDataAccessGroups" => "false",
	    "returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	record = HTTP.post(ENV["REDCap_url"], body=HTTP.Form(rec_payload), verbose = true)

	# Parse the JSON response
	record = JSON.parse(String(record.body))


	# Get the files
	jspsych_data = []
	for r in record
		if r["other_data"] == "file"
			tdata = get_REDCap_file(r["record_id"])

			# Add record_id
			for tr in tdata
				tr["record_id"] = r["record_id"]
			end
			
			push!(jspsych_data, tdata)
		end
	end

	
	return jspsych_data, record
end

# Convert to df and merge REDCap record data and jsPsych data
function REDCap_data_to_df(jspsych_data, records)
	
	# Records to df
	records_df = DataFrame(records)
	
	# Convert to DataFrame
	jspsych_data = reduce(vcat, jspsych_data)

	jspsych_data = vcat(
		[DataFrame(d) for d in jspsych_data]...,
		cols=:union
	)

	jspsych_data = leftjoin(jspsych_data, 
		rename(records_df[!, [:prolific_pid, :record_id, :start_time]],
			:start_time => :exp_start_time),
		on = [:prolific_pid, :record_id]
	)

	return jspsych_data
end

remove_testing!(data::DataFrame) = filter!(x -> !occursin(r"yaniv|tore|demo", x.prolific_pid), data)

# Filter PLT data
function prepare_PLT_data(data::DataFrame)

	# Select rows
	PLT_data = filter(x -> x.trial_type == "PLT", data)

	# Select columns
	PLT_data = PLT_data[:, Not(map(col -> all(ismissing, col), eachcol(PLT_data)))]

	# Filter practice
	filter!(x -> typeof(x.block) == Int64, PLT_data)

	# Sort
	sort!(PLT_data, [:prolific_pid, :session, :block, :trial])

	return PLT_data

end

# Load PLT data from file or REDCap
function load_PLT_data()
	datafile = "data/data.jld2"
	if !isfile(datafile)
		jspsych_data, records = get_REDCap_data()
		
		data = REDCap_data_to_df(jspsych_data, records)
		
		remove_testing!(data)

		JLD2.@save datafile data
	else
		JLD2.@load datafile data
	end
	
	PLT_data = prepare_PLT_data(data)

    return PLT_data
end

# Exclude unfinished and double sessions
function exclude_PLT_sessions(PLT_data::DataFrame)
	# Find non-finishers
	non_finishers = combine(groupby(PLT_data,
		[:prolific_pid, :session, 
		:exp_start_time, :condition]),
		:block => (x -> length(unique(x))) => :n_blocks
	)

	filter!(x -> x.n_blocks < 24, non_finishers)

	# Exclude non-finishers
	PLT_data_clean = antijoin(PLT_data, non_finishers,
		on = [:prolific_pid, :session, 
		:exp_start_time, :condition])

	# Find double takes
	double_takers = unique(PLT_data_clean[!, [:prolific_pid, :session, 
		:exp_start_time, :condition]])

	# Find earliert session
	double_takers.date = DateTime.(double_takers.exp_start_time, 
		"yyyy-mm-dd_HH:MM:SS")

	DataFrames.transform!(
		groupby(double_takers, [:prolific_pid, :session]),
		:condition => length => :n,
		:date => minimum => :first_date
	)

	filter!(x -> (x.n > 1) & (x.date != x.first_date), double_takers)

	# Exclude extra sessions
	PLT_data_clean = antijoin(PLT_data_clean, double_takers,
		on = [:prolific_pid, :session, 
		:exp_start_time, :condition]
	)

	return PLT_data_clean

end

function exclude_PLT_trials(PLT_data::DataFrame)

	# Exclude missed responses
	PLT_data_clean = filter(x -> x.choice != "noresp", PLT_data)

	return PLT_data_clean
end

# Function for computing number of consecutive optimal chioces
function count_consecutive_ones(v)
	# Initialize the result vector with the same length as v
	result = zeros(Int, length(v))
	# Initialize the counter
	counter = 0

	for i in 1:length(v)
		if v[i] == 1
			# Increment the counter if the current element is 1
			counter += 1
		else
			# Reset the counter to 0 if the current element is 0
			counter = 0
		end
		# Store the counter value in the result vector
		result[i] = counter
	end

	return result
end

"""
    task_vars_for_condition(condition::String)

Fetches and processes the task structure for a given experimental condition, preparing key variables for use in reinforcement learning models.

# Arguments
- `condition::String`: The specific condition for which the task structure is to be retrieved. This string is used to load the appropriate CSV file corresponding to the condition.

# Returns
- A named tuple with the following components:
  - `task`: A `DataFrame` containing the full task structure loaded from a CSV file, with processed block numbers and feedback columns.
  - `block`: A vector of block numbers adjusted to account for session numbers, useful for tracking the progression of blocks across multiple sessions.
  - `valence`: A vector containing the unique valence values for each block, indicating the nature of feedback (e.g., positive or negative) associated with each block.
  - `outcomes`: A matrix where the first column contains feedback for the suboptimal option and the second column contains feedback for the optimal option. This arrangement is designed to facilitate learning model implementations where the optimal outcome is consistently in the second column.

# Details
- The task structure is loaded from a CSV file named `"PLT_task_structure_$condition.csv"` located in the `data` directory, where `$condition` is replaced by the value of the `condition` argument.
- Block numbers are renumbered to reflect their session, allowing for consistent tracking across multiple sessions.
- Feedback values are reorganized based on the optimal choice (either option A or B), with the optimal feedback placed in one column and the suboptimal feedback in the other.
- This function is useful for preparing task-related variables for reinforcement learning models that require specific input formats.
"""
function task_vars_for_condition(condition::String)
	# Load sequence from file
	task = DataFrame(CSV.File("data/PLT_task_structure_$condition.csv"))

	# Renumber block
	task.block = task.block .+ (task.session .- 1) * maximum(task.block)

	# Arrange feedback by optimal / suboptimal
	task.feedback_optimal = 
		ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

	task.feedback_suboptimal = 
		ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)


	# Arrange outcomes such as second column is optimal
	outcomes = hcat(
		task.feedback_suboptimal,
		task.feedback_optimal,
	)

	return (
		task = task,
		block = task.block,
		valence = unique(task[!, [:block, :valence]]).valence,
		outcomes = outcomes
	)

end