# Functions for fetching data and preprocessing it

# Fetch one data file by record_id
function get_REDCap_file(
	record_id::String;
	experiment::String,
	field::String = "other_data"
)
	# Create the payload for getting the file
	file_payload = Dict(
		"token" => ENV["$(experiment)_REDCap_token"],
	    "content" => "file",
		"action" => "export",
		"record" => record_id,
		"field" => field,
		"returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	file = HTTP.post(ENV["REDCap_url"], body=HTTP.Form(file_payload), verbose = true)

	# Parse
	return JSON.parse(String(file.body))
end

# Fetch entire dataset
function get_REDCap_data(
	experiment::String;
	file_field::String = "other_data" # Field on REDCap database containing task data file
	)

	# Get the records --------
	# Create the payload for getting the record details
	rec_payload = Dict(
		"token" => ENV["$(experiment)_REDCap_token"],
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
		if r[file_field] == "file"
			tdata = get_REDCap_file(r["record_id"]; 
				experiment = experiment, 
				field = file_field
			)

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
	
	# Concatenate trials
	jspsych_data = reduce(vcat, jspsych_data)

	# Helper function to jsonify arrays and dicts
	function jsonify_multidimensional_data(d::Dict)
	    for (k, v) in d
	        if (v isa AbstractArray) || (v isa Dict)
				d[k] = json(v)
			end
	    end
	    return d
	end

	# Convert to DataFrame
	jspsych_data = vcat(
		[DataFrame(jsonify_multidimensional_data(d)) for d in jspsych_data]...,
		cols=:union
	)

	# Exclude missing prolific_pid
	filter!(x -> !ismissing(x.prolific_pid), jspsych_data)

	# Combine records and jspsych data
	jspsych_data = leftjoin(jspsych_data, 
		rename(records_df[!, [:prolific_pid, :record_id, :start_time]],
			:start_time => :exp_start_time),
		on = [:prolific_pid, :record_id]
	)

	return jspsych_data
end

remove_testing!(data::DataFrame) = filter!(x -> (!occursin(r"haoyang|yaniv|tore|demo|simulate|debug", x.prolific_pid)) && (length(x.prolific_pid) > 10), data)

# Filter PLT data
function prepare_PLT_data(data::DataFrame; trial_type::String = "PLT")

	# Select rows
	PLT_data = filter(x -> x.trial_type == trial_type, data)

	# Select columns
	PLT_data = PLT_data[:, Not(map(col -> all(ismissing, col), eachcol(PLT_data)))]

	# Filter practice
	filter!(x -> typeof(x.block) == Int64, PLT_data)

	# Sort
	sort!(PLT_data, [:prolific_pid, :session, :block, :trial])

	return PLT_data

end

function load_pilot6_data(; force_download = false, return_version = "6.01")
	datafile = "data/pilot6.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot6"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		filter!(x -> x.version ∈ ["6.0", "6.01"], jspsych_data)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	remove_testing!(jspsych_data)

	# Subset version for return
	filter!(x -> x.version == return_version, jspsych_data)

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Divide intwo WM and PILT
	WM_data = filter(x -> x.n_stimuli == 3, PILT_data)
	filter!(x -> x.n_stimuli == 2, PILT_data)

	# Extract post-PILT test
	test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	return PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data, reversal_data, jspsych_data
end

function load_pilot4x_data(; force_download = false)
	datafile = "data/pilot4.x.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot4.x"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PLT_data = prepare_PLT_data(jspsych_data)

	# Extract post-PILT test
	test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	return PLT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, reversal_data, jspsych_data
end


function load_pilot4_data()
	datafile = "data/pilot4.jld2"

	# Load data or download from REDCap
	if !isfile(datafile)
		jspsych_json, records = get_REDCap_data("pilot4"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PLT_data = prepare_PLT_data(jspsych_data)

	# Extract post-PILT test
	test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)

	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	return PLT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, reversal_data, jspsych_data
end


function load_pilot2_data()
	datafile = "data/pilot2.jld2"

	# Load data or download from REDCap
	if !isfile(datafile)
		jspsych_json, records = get_REDCap_data("pilot2"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract three tasks
	PLT_data = prepare_PLT_data(jspsych_data)

	test_data = prepare_post_PILT_test_data(jspsych_data)

	### Vigour task here
	vigour_data = prepare_vigour_data(jspsych_data) |>
		x -> exclude_vigour_trials(x, 66)

	return PLT_data, test_data, vigour_data, jspsych_data
end

# Load PLT data from file or REDCap
function load_pilot1_data()
	datafile = "data/pilot1.jld2"
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

function exclude_double_takers!(df::DataFrame)
	# Find double takes
	double_takers = unique(df[!, [:prolific_pid, :session, 
		:exp_start_time]])

	# Function to parse date with multiple formats (WorldClock API format and jsPsych format)
	function parse_date(date_str)
		# If date_str ends with "Z", replace it with "+00:00" for proper parsing
		if endswith(date_str, "Z")
			date_str = replace(date_str, "Z" => "")
		end
		
		for fmt in ["yyyy-mm-dd_HH:MM:SS", "yyyy-mm-ddTHH:MM:SS.ssszzzz", "yyyy-mm-ddTHH:MM:SS"]
			try
				return DateTime(date_str, fmt)
			catch
				# Ignore and try the next format
			end
		end
		
		error("Date format not recognized: $date_str")
	end

	# Find earliert session
	double_takers.date = parse_date.(double_takers.exp_start_time)

	DataFrames.DataFrames.transform!(
		groupby(double_takers, [:prolific_pid, :session]),
		:session => length => :n,
		:date => minimum => :first_date
	)

	filter!(x -> (x.n > 1) & (x.date != x.first_date), double_takers)

	# Exclude extra sessions
	df = antijoin(df, double_takers,
		on = [:prolific_pid, :session, 
		:exp_start_time]
	)
end

# Exclude unfinished and double sessions
function exclude_PLT_sessions(PLT_data::DataFrame; required_n_blocks::Int64 = 24)
	# Find non-finishers
	non_finishers = combine(groupby(PLT_data,
		[:prolific_pid, :session, 
		:exp_start_time]),
		:block => (x -> length(unique(x))) => :n_blocks
	)

	filter!(x -> x.n_blocks < required_n_blocks, non_finishers)

	# Exclude non-finishers
	PLT_data_clean = antijoin(PLT_data, non_finishers,
		on = [:prolific_pid, :session, 
		:exp_start_time])

	exclude_double_takers!(PLT_data_clean)

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
    prepare_post_PILT_test_data(data::AbstractDataFrame) -> DataFrame

Processes and prepares data from the PILT test phase for further analysis. This function filters rows to include only those from the PILT test phase, removes columns where all values are missing, and computes additional columns based on participant responses.

# Arguments
- `data::AbstractDataFrame`: The raw experimental data, including trial phases and participant responses.

# Returns
- A DataFrame with the PILT test data, including computed columns for the chosen stimulus and whether the response was "ArrowRight". Columns with all missing values are excluded.
"""
function prepare_post_PILT_test_data(data::AbstractDataFrame)

	# Select rows
	test_data = filter(x -> !ismissing(x.trialphase) && (x.trialphase == "PILT_test"), data)

	# Select columns
	test_data = test_data[:, Not(map(col -> all(ismissing, col),
		eachcol(test_data)))]

	if :stimulus in names(test_data)
		select!(test_data, Not(:stimulus))
	end

	# Compute chosen stimulus
	@assert Set(test_data.response) ⊆ Set(["ArrowRight", "ArrowLeft", "null", nothing]) "Unexected responses in PILT test data"
	
	test_data.chosen_stimulus = ifelse.(
		test_data.response .== "ArrowRight",
		test_data.stimulus_right,
		ifelse.(
			test_data.response .== "ArrowLeft",
			test_data.stimulus_left,
			missing
		)
	)

	test_data.right_chosen = (x -> get(Dict("ArrowRight" => true, "ArrowLeft" => false, "null" => missing), x, missing)).(test_data.response)

	return test_data

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
- The task structure is loaded from a CSV file named `"PLT_task_structure_\$condition.csv"` located in the `data` directory, where `\$condition` is replaced by the value of the `condition` argument.
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

function safe_mean(arr)
	if ismissing(arr) || isempty(arr)  # Check if array is missing or empty
			return missing
	elseif all(x -> x isa Number, arr)  # Check if all elements are numeric
			return mean(arr)
	else
			return missing  # Return missing if the array contains non-numeric elements
	end
end

function safe_median(arr)
	if ismissing(arr) || isempty(arr)  # Check if array is missing or empty
			return missing
	elseif all(x -> x isa Number, arr)  # Check if all elements are numeric
			return median(arr)
	else
			return missing  # Return missing if the array contains non-numeric elements
	end
end

function prepare_vigour_data(data::DataFrame)
	# Define required columns for vigour data
	required_columns = [:prolific_pid, :record_id, :version, :exp_start_time, :trialphase, :trial_number, :trial_duration, :response_time, :timeline_variables]
	required_columns = vcat(required_columns, names(data, r"(reward|presses)$"))

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(data))
            insertcols!(data, col => missing)
        end
    end

	# Prepare vigour data
	vigour_data = data |>
		x -> select(x, 
			:prolific_pid,
			:record_id,
			:version,
			:exp_start_time,
			:session,
			:trialphase,
			:trial_number,
			:trial_duration,
			names(x, r"(reward|presses)$"),
			:response_time,
			:timeline_variables
		) |>
		x -> subset(x, 
	        [:trialphase, :trial_number] => ByRow((x, y) -> (!ismissing(x) && x in ["vigour_trial"]) || (!ismissing(y)))
	    ) |>
	  	x -> DataFrames.transform(x,
			:response_time => ByRow(JSON.parse) => :response_times,
			:timeline_variables => ByRow(x -> JSON.parse(x)["ratio"]) => :ratio,
			:timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]) => :magnitude,
			:timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]/JSON.parse(x)["ratio"]) => :reward_per_press
		) |>
		x -> select(x, 
			Not([:response_time, :timeline_variables])
		)
		vigour_data = exclude_double_takers!(vigour_data)
	return vigour_data
end

function prepare_post_vigour_test_data(data::DataFrame)
	# Define required columns for vigour data
	required_columns = [:prolific_pid, :record_id, :version, :exp_start_time, :trialphase, :response]
	required_columns = vcat(required_columns, names(data, r"(magnitude|ratio)$"))

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(data))
            insertcols!(data, col => missing)
        end
    end

	# Prepare post vigour test data
	post_vigour_test_data = data |>
		x -> select(x,
			:prolific_pid,
			:session,
		    :record_id,
		    :version,
		    :exp_start_time,
		    :trialphase,
		    :response,
			:rt => :response_times,
		    r"magnitude$",
		    r"ratio$"
		) |>
		x -> subset(x, :trialphase => ByRow(x -> !ismissing(x) && x in ["vigour_test"])) |>
		x -> groupby(x, [:prolific_pid, :exp_start_time]) |>
		x -> DataFrames.transform(x, :trialphase => (x -> 1:length(x)) => :trial_number)
		post_vigour_test_data = exclude_double_takers!(post_vigour_test_data)
	return post_vigour_test_data
end

function prepare_PIT_data(data::DataFrame)
	
	# Define required columns for vigour data
	required_columns = [:prolific_pid, :record_id, :version, :exp_start_time, :trialphase, :pit_trial_number, :trial_duration, :response_time, :pit_coin, :timeline_variables]
	required_columns = vcat(required_columns, names(data, r"(reward|presses)$"))

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(data))
            insertcols!(data, col => missing)
        end
    end

	# Prepare PIT data
	PIT_data = data |>
		x -> select(x, 
			:prolific_pid,
			:session,
			:record_id,
			:version,
			:exp_start_time,
			:trialphase,
			:pit_trial_number => :trial_number,
			:trial_duration,
			names(x, r"(reward|presses)$"),
			:response_time,
			:pit_coin => :coin,
			:timeline_variables
		) |>
		x -> subset(x, 
	        :trialphase => ByRow(x -> !ismissing(x) && x in ["pit_trial"])
	    ) |>
	  	x -> DataFrames.transform(x,
			:response_time => ByRow(JSON.parse) => :response_times,
			:timeline_variables => ByRow(x -> JSON.parse(x)["ratio"]) => :ratio,
			:timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]) => :magnitude,
			:timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]/JSON.parse(x)["ratio"]) => :reward_per_press
		) |>
		x -> select(x, 
			Not([:response_time, :timeline_variables])
		)
		PIT_data = exclude_double_takers!(PIT_data)
	return PIT_data
end


"""
	exclude_vigour_trials(vigour_data::DataFrame) -> DataFrame

This function processes the given `vigour_data` DataFrame to exclude certain trials based on specific criteria:

1. **Non-finishers**: Participants who have completed fewer than complete trials are identified and excluded.
2. **Double takes**: Participants who have multiple sessions are identified, and only the earliest session is retained.

# Arguments
- `vigour_data::DataFrame`: The input DataFrame containing vigour trial data.

# Returns
- `DataFrame`: A cleaned DataFrame with non-finishers and extra trials from multiple sessions excluded.
"""
function exclude_vigour_trials(vigour_data::DataFrame, n_trials::Int)
	# Find non-finishers
	non_finishers = combine(groupby(vigour_data,
		[:prolific_pid, :exp_start_time]),
		:trial_number => (x -> length(unique(x))) => :n_trials
	)

	filter!(x -> x.n_trials < n_trials, non_finishers)

	# Exclude non-finishers
	vigour_data_clean = antijoin(vigour_data, non_finishers,
		on = [:prolific_pid, :exp_start_time])

	# Find double takes
	double_takers = unique(vigour_data_clean[!, [:prolific_pid, :exp_start_time]])

	# Find earliert session
	double_takers.date = DateTime.(double_takers.exp_start_time, 
		"yyyy-mm-dd_HH:MM:SS")

	DataFrames.transform!(
		groupby(double_takers, [:prolific_pid]),
		:date => minimum => :first_date
	)

	filter!(x -> x.date != x.first_date, double_takers)

	# Exclude extra trials from multiple participants
	vigour_data_clean = antijoin(vigour_data_clean, double_takers,
		on = [:prolific_pid, :exp_start_time]
	)

	return vigour_data_clean
end

"""
    prepare_reversal_data(data::DataFrame) -> DataFrame

Extracts reversal task data. The function removes columns where all values are missing, sorts the remaining data by `:prolific_pid`, `:session`, `:block`, and `:trial`, and returns the cleaned DataFrame.

# Arguments
- `data::DataFrame`: The input DataFrame containing experimental data.

# Returns
- `DataFrame`: A filtered and sorted DataFrame containing only reversal trials.
"""
function prepare_reversal_data(data::DataFrame)
	reversal_data = filter(x -> x.trial_type == "reversal", data)

	# Select columns
	reversal_data = reversal_data[:, Not(map(col -> all(ismissing, col), eachcol(reversal_data)))]

	# Sort
	sort!(reversal_data, [:prolific_pid, :session, :block, :trial])

	return reversal_data
end

function exclude_reversal_sessions(
	reversal_data::DataFrame;
	required_n_trials::Int64 = 200
)
	non_finishers = combine(
		groupby(reversal_data, [:prolific_pid, :session, 
		:exp_start_time]),
		:trial => length => :n_trials
	)

	filter!(x -> x.n_trials < required_n_trials,  non_finishers)

	# Exclude non-finishers
	reversal_data_clean = antijoin(reversal_data, non_finishers,
		on = [:prolific_pid, :session, 
		:exp_start_time])


	exclude_double_takers!(reversal_data_clean)

	return reversal_data_clean
end