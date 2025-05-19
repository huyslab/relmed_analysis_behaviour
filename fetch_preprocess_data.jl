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
	file_field::String = "other_data", # Field on REDCap database containing task data file
	record_id_field::String = "record_id" # Field on REDCap database containing record ID
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
	# Rename participant_id to record_id in each record
	map(x -> x["record_id"] = x[record_id_field], record)

	# Get the files
	jspsych_data = []
	for r in record
		if r[file_field] == "file"
			tdata = get_REDCap_file(r[record_id_field]; 
				experiment = experiment, 
				field = file_field
			)

			# Add record_id
			for tr in tdata
				tr["record_id"] = r[record_id_field]
			end
			
			push!(jspsych_data, tdata)
		end
	end

	
	return jspsych_data, record
end

# Convert to df and merge REDCap record data and jsPsych data
function REDCap_data_to_df(jspsych_data, records; participant_id_field::String = "prolific_pid", start_time_field::String = "start_time")
	
	# Convert to symbols
	participant_id = Symbol(participant_id_field) # Prolific ID if available
	start_time = Symbol(start_time_field)

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
	filter!(x -> !ismissing(x[participant_id]), jspsych_data)

	# Needed variables from records_df
	col_names = [participant_id_field, "record_id", start_time_field];
	valid_cols = intersect(col_names, names(records_df))
	records_df = records_df[!, valid_cols]

	# Where to find the start_time field
	if start_time_field in valid_cols
		@info "start_time field found in records_df"
		rename!(records_df, start_time => :exp_start_time)
	end
	if start_time_field in names(jspsych_data)
		@info "start_time field found in jspsych_data"
		rename!(jspsych_data, start_time => :exp_start_time)
	end

	# Join by variables
	on_cols = intersect(names(jspsych_data), names(records_df))
	@info "Joining on columns: $on_cols"
	# Combine records and jspsych data
	jspsych_data = leftjoin(
		jspsych_data, records_df,
		on = map(Symbol, on_cols)
	)

	transform!(jspsych_data, participant_id => :prolific_pid)

	return jspsych_data
end

remove_testing!(data::DataFrame) = filter!(x -> (!occursin(r"haoyang|yaniv|tore|demo|simulate|debug", x.prolific_pid)) && (length(x.prolific_pid) > 10), data)

# Filter PLT data
function prepare_PLT_data(data::DataFrame; trial_type::String = "PLT")

	# Select rows
	PLT_data = filter(x -> (x.trial_type == trial_type) && (x.trialphase != "PILT_test"), data)

	# Select columns
	PLT_data = PLT_data[:, Not(map(col -> all(ismissing, col), eachcol(PLT_data)))]

	# Filter practice
	filter!(x -> typeof(x.block) == Int64, PLT_data)

	# Sort
	sort!(PLT_data, [:prolific_pid, :session, :block, :trial])

	return PLT_data

end

function load_control_pilot2_data(; force_download = false, session = "1")
	datafile = "data/control_pilot2.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("control_pilot2"; file_field = "data", record_id_field = "record_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "sitting_start_time")

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end
	# Subset version for return
	# filter!(x -> x.version == return_version, jspsych_data)
	filter!(x -> x.session == session, jspsych_data)

	# Extract control data
	control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return control_task_data, control_report_data, jspsych_data
end




function load_trial1_data(; force_download = false)
	datafile = "data/trial1.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("testing"; file_field = "jspsych_data", record_id_field = "record_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "sitting_start_time")

		# remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Extract WM data
	WM_data = filter(x -> x.trialphase == "wm", PILT_data)

	# Extract LTM data
	LTM_data = filter(x -> x.trialphase == "ltm", PILT_data)

	# Extract WM test data
	WM_test_data = filter(x -> x.trialphase == "wm_test", PILT_data)

	# Extract LTM test data
	LTM_test_data = filter(x -> x.trialphase == "ltm_test", PILT_data)

	# Seperate out PILT
	filter!(x -> x.trialphase == "pilt", PILT_data)
	
	# Extract post-PILT test
	# test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	# vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	# post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)
			
	# Extract PIT
	# PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	reversal_data = prepare_reversal_data(jspsych_data)

	# Extract max press rate data
	# max_press_data = prepare_max_press_data(jspsych_data)

	# Extract control data
	# control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return PILT_data, WM_data, LTM_data, reversal_data, jspsych_data
end

function load_pilot9_data(; force_download = false)
	datafile = "data/pilot9.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot9"; file_field = "data", record_id_field = "record_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "sitting_start_time")

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Seperate out PILT
	filter!(x -> x.trialphase == "pilt", PILT_data)
	
	# Extract post-PILT test
	test_data = prepare_test_data(jspsych_data)

	# Exctract vigour
	vigour_data = prepare_vigour_data(jspsych_data) 
			
	# Extract PIT
	PIT_data = prepare_PIT_data(jspsych_data)

	# Extract max press rate data
	# max_press_data = prepare_max_press_data(jspsych_data)

	# Extract control data
	# control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return PILT_data, test_data, vigour_data, PIT_data, jspsych_data
end

function load_pilot8_data(; force_download = false, return_version = "0.2")
	datafile = "data/pilot8.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot8"; file_field = "file_data", record_id_field = "participant_id")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records; participant_id_field = "participant_id", start_time_field = "module_start_time")

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

	# Subset version for return
	filter!(x -> x.version == return_version, jspsych_data)

	# Exctract PILT
	PILT_data = prepare_PLT_data(jspsych_data; trial_type = "PILT")

	# Extract WM data
	WM_data = filter(x -> x.trialphase == "wm", PILT_data)

	# Extract LTM data
	LTM_data = filter(x -> x.trialphase == "ltm", PILT_data)

	# Extract WM test data
	WM_test_data = filter(x -> x.trialphase == "wm_test", PILT_data)

	# Extract LTM test data
	LTM_test_data = filter(x -> x.trialphase == "ltm_test", PILT_data)

	# Seperate out PILT
	filter!(x -> x.trialphase == "pilt", PILT_data)
	
	# Extract post-PILT test
	# No post-PILT test in Pilot8
	# test_data = prepare_post_PILT_test_data(jspsych_data)

	# Exctract vigour
	# No vigour task in Pilot8
	# vigour_data = prepare_vigour_data(jspsych_data) 

	# Extract post-vigour test
	# post_vigour_test_data = prepare_post_vigour_test_data(jspsych_data)
			
	# Extract PIT
	# No PIT task in Pilot8
	# PIT_data = prepare_PIT_data(jspsych_data)

	# Exctract reversal
	# No reversal task in Pilot7
	# reversal_data = prepare_reversal_data(jspsych_data)

	# Extract max press rate data
	max_press_data = prepare_max_press_data(jspsych_data)

	# Extract control data
	control_task_data, control_report_data = prepare_control_data(jspsych_data) 

	return PILT_data, WM_data, LTM_data, WM_test_data, LTM_test_data, max_press_data, control_task_data, control_report_data, jspsych_data
end

function load_pilot7_data(; force_download = false, return_version = "6.01")
	datafile = "data/pilot7.jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
		jspsych_json, records = get_REDCap_data("pilot7"; file_field = "file_data")
	
		jspsych_data = REDCap_data_to_df(jspsych_json, records)

		remove_testing!(jspsych_data)

		JLD2.@save datafile jspsych_data
	else
		JLD2.@load datafile jspsych_data
	end

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
	# No reversal task in Pilot7
	# reversal_data = prepare_reversal_data(jspsych_data)

	# Extract max press rate data
	max_press_data = prepare_max_press_data(jspsych_data)

	return PILT_data, test_data, vigour_data, post_vigour_test_data, PIT_data, WM_data, max_press_data, jspsych_data
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

function exclude_double_takers(df::DataFrame)
	# Find double takes
	double_takers = unique(df[!, [:prolific_pid, :session, :exp_start_time]])

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

	DataFrames.transform!(
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

	return df
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

	return exclude_double_takers(PLT_data_clean)
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

function prepare_test_data(df::DataFrame; task::String = "pilt")

	# Select rows
	test_data = filter(x -> (x.trial_type == "PILT") && (x.trialphase == "$(task)_test"), df)

	# Select columns
	test_data = test_data[:, Not(map(col -> all(ismissing, col), eachcol(test_data)))]

	# Change all block names to same type
	test_data.block = string.(test_data.block)

	# Sort
	sort!(test_data, [:participant_id, :session, :block, :trial])

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

function prepare_max_press_data(data::DataFrame)
	# Define required columns for vigour data
	required_columns = [:prolific_pid, :record_id, :version, :exp_start_time, :session, :trialphase, :trial_number, :avgSpeed, :responseTime, :trialPresses]

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(data))
            insertcols!(data, col => missing)
        end
    end

	# Prepare vigour data
	max_press_data = data |>
		x -> filter(x -> !ismissing(x.trialphase) && x.trialphase == "max_press_rate", x) |>
		x -> select(x, 
			:prolific_pid,
			:record_id,
			:version,
			:exp_start_time,
			:session,
			:trialphase,
			:trial_number,
			# :trial_duration,
			:avgSpeed => :avg_speed,
			:responseTime,
			:trialPresses => :trial_presses
		) |>
		x -> subset(x, 
				[:trialphase, :trial_number] => ByRow((x, y) -> (!ismissing(x) && x in ["max_press_rate"]) || (!ismissing(y)))
		) |>
		x -> DataFrames.transform(x,
			:responseTime => ByRow(JSON.parse) => :response_times
		) |>
		x -> select(x, 
			Not([:responseTime])
		)
		max_press_data = exclude_double_takers(max_press_data)
	return max_press_data
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
		vigour_data = exclude_double_takers(vigour_data)
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
		post_vigour_test_data = exclude_double_takers(post_vigour_test_data)
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
		PIT_data = exclude_double_takers(PIT_data)
	return PIT_data
end

function get_chosen_stimulus(row)
    if row.response == "left"
        return row.stimulus_left
    elseif row.response == "right"
        return row.stimulus_right
    else # "middle"
        return row.stimulus_middle
    end
end

function get_choice_number(row, block_stimuli_map)
    # Use compound key
    key = (row.block, row.stimulus_group_id, row.session)
    stimulus_to_number = block_stimuli_map[key]
    chosen = get_chosen_stimulus(row)
    return stimulus_to_number[chosen]
end

function add_choice_column!(df)
    # Map (block, stimulus_group_id) to stimulus mappings
    block_stimuli_map = Dict()
    
    for block_grp in unique(zip(df.block, df.stimulus_group_id, df.session))
        block, grp, sess = block_grp
        block_data = filter(
			r -> r.block == block && r.stimulus_group_id == grp && r.session == sess, df
		)
        
        stimuli = unique([
            block_data.stimulus_left;
            block_data.stimulus_right;
            block_data.stimulus_middle
        ])
        
        optimal = first(filter(r -> r.optimal_side == "middle", block_data)).stimulus_middle
        others = setdiff(stimuli, [optimal])
        
        stimulus_to_number = Dict(
            optimal => 3,
            others[1] => 1,
            others[2] => 2
        )
        
        block_stimuli_map[block_grp] = stimulus_to_number
    end
    
    df.choice = [get_choice_number(row, block_stimuli_map) for row in eachrow(df)]
    return df
end

# Prepare pilot data for fititng with model
function prepare_WM_data(data)
	forfit = exclude_PLT_sessions(data, required_n_blocks = 10)
	filter!(x -> x.response != "noresp", forfit)
	forfit.feedback_optimal = 
		ifelse.(
			forfit.optimal_side .== "middle",
			forfit.feedback_middle,
			ifelse.(
				forfit.optimal_side .== "left", forfit.feedback_left, forfit.feedback_right
			)
		)
	
	# Suboptimal feedback is always the same, so can be copied
	forfit.feedback_suboptimal1 = 
		ifelse.(
			forfit.optimal_side .== "middle",
			forfit.feedback_left,
			ifelse.(forfit.optimal_side .== "left", forfit.feedback_right, forfit.feedback_middle)
		)
	forfit.feedback_suboptimal2 = forfit.feedback_suboptimal1

	# Make choice into an integer column
	forfit = add_choice_column!(forfit)

	# Clean block numbers up
	renumber_block(x) = indexin(x, sort(unique(x)))
    DataFrames.transform!(
		groupby(forfit, [:prolific_pid, :session]),
		:block => renumber_block => :block,
		ungroup=true
	)
	forfit.block = convert(Vector{Int64}, forfit.block)

	# PID as number
	pids = unique(forfit[!, [:prolific_pid]])
	pids.PID = 1:nrow(pids)
	forfit = innerjoin(forfit, pids[!, [:prolific_pid, :PID]], on = :prolific_pid)

	# New columns for compatilibility with WM functions
	forfit.set_size = forfit.n_groups .* 3
	forfit.pair = forfit.stimulus_group
	forfit.isOptimal = forfit.response_optimal
	forfit = DataFrames.transform(
		groupby(forfit, [:PID, :block, :pair, :session]), eachindex => :trial; ungroup=true
	)

	return forfit
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


	return exclude_double_takers(reversal_data_clean)
end

"""
	extract_timeline_variables!(df::DataFrame) -> DataFrame

Extract and flatten JSON data from the `:timeline_variables` column of a DataFrame.

This function parses the JSON data stored in the `:timeline_variables` column,
creates new columns in the DataFrame for each unique key found in the JSON objects,
and then removes the original `:timeline_variables` column.

# Arguments
- `df::DataFrame`: A DataFrame containing a `:timeline_variables` column with JSON data.

# Returns
- The modified DataFrame with JSON data flattened into separate columns.

# Note
- The function modifies the input DataFrame in-place.
- If a row's `:timeline_variables` field is missing or cannot be parsed as JSON, 
	it will be treated as an empty dictionary.
- If a row's JSON doesn't contain a particular key found in other rows, 
	the corresponding cell will be assigned `missing`.
"""
function extract_timeline_variables!(df::DataFrame)
	parsed = map(row -> begin
			ismissing(row.timeline_variables) && return Dict()
			str = startswith(row.timeline_variables, "{") ? row.timeline_variables : "{" * row.timeline_variables
			try JSON.parse(str) catch; Dict() end
	end, eachrow(df))
	
	for key in unique(Iterators.flatten(keys.(parsed)))
			df[!, key] = [get(p, key, missing) for p in parsed]
	end

	select!(df, Not(:timeline_variables))
	
	return df
end

"""
	merge_control_task_and_feedback(df_a, df_b)

Merge two dataframes containing experimental data, typically from control task and feedback phases.

# Arguments
- `df_a`: Primary dataframe (likely containing control task data)
- `df_b`: Secondary dataframe (likely containing feedback data)

# Returns
A merged dataframe that:
- Joins data on keys: "exp_start_time", "prolific_pid", "record_id", "session", "task", "trial"
- Removes redundant variables from both dataframes
- Preserves specific variables only from dataframe A (time_elapsed, trialphase, etc.)
- Handles the "correct" variable specially by combining values from both sources
- Returns the result sorted by record_id and trial

# Details
The function performs a left join, keeping all rows from df_a and matching rows from df_b.
Certain variables are intentionally excluded to avoid redundancy or kept exclusively from
the first dataframe to maintain data consistency. If both dataframes contain a "correct" 
variable, the values are coalesced with preference given to df_a's values.
"""
function merge_control_task_and_feedback(df_a, df_b)
	# 1. Identify the key variables to join on
	# join_keys = ["exp_start_time", "prolific_pid", "record_id", "session", "task", "trial"]
	
	# 2. Variables to remove from both dataframes before merging
	remove_vars = ["n_warnings", "plugin_version", "pre_kick_out_warned", "trial_index", "trial_type", "version", "trial_ptype"]
	
	# 3. Variables to keep only from dataframe A and not from B
	#  "current_strength", "effort_level", "near_island" are redundant since they are included in the timeline variables
	keep_from_a = ["time_elapsed", "trialphase", "current_strength", "effort_level", "near_island"]
	
	# 4. Special handling for "correct" (will be combined)
	
	# Create copies to avoid modifying originals
	df_a_clean = select(df_a, Not(remove_vars))
	df_b_clean = select(df_b, Not(intersect(names(df_b), vcat(remove_vars, keep_from_a))))
	
	# Rename the "correct" column in df_b to avoid collision
	if "correct" in names(df_b_clean)
			rename!(df_b_clean, "correct" => "correct_b")
	end
	
	# Merge the dataframes
	join_keys = intersect(names(df_a_clean), names(df_b_clean))
	merged_df = leftjoin(df_a_clean, df_b_clean, on=join_keys)
	
	# Handle the "correct" variable - combining both sources
	if "correct" in names(merged_df) && "correct_b" in names(merged_df)
			# Create a new combined "correct" column
			# This assumes you want to use df_a's value if available, otherwise df_b's
	merged_df.combined_correct = coalesce.(merged_df.correct, merged_df.correct_b)
			
			# Remove the original columns and rename the combined one
			select!(merged_df, Not([:correct, :correct_b]))
			rename!(merged_df, :combined_correct => :correct)
	end
	
	return sort(merged_df, [:record_id, :trial])
end

"""
	prepare_control_data(data::DataFrame) -> Tuple{DataFrame, DataFrame}

Preprocess experimental data to extract and format control-related trial information.

# Arguments
- `data::DataFrame`: Raw experimental data containing all trial types.

# Processing Steps
1. Filters data to include only control-related trial phases
2. Removes participants who completed the experiment multiple times
3. Drops columns with all missing values
4. Creates trial numbering for each participant
5. Separates data into task, feedback, and report components
6. Extracts and parses timeline variables and response times
7. Merges task data with corresponding feedback data

# Returns
A tuple containing two DataFrames:
- `control_task_data`: Processed data from main control trials with feedback information
- `control_report_data`: Participant's confidence and controllability reports

# Note
This function expects certain column names to be present in the input DataFrame,
including 'trialphase', 'record_id', 'trial_index', 'responseTime', etc.
"""
function prepare_control_data(data::DataFrame)
	control_data = filter(x -> !ismissing(x.trialphase) && x.trialphase ∈ ["control_explore", "control_explore_feedback", "control_controllability", "control_predict_homebase", "control_confidence", "control_reward", "control_reward_feedback"], data)
	control_data = exclude_double_takers(control_data)
	
	for col in names(control_data)
		control_data[!, col] = [val === nothing ? missing : val for val in control_data[!, col]]
	end
	control_data = control_data[:, .!all.(ismissing, eachcol(control_data))]
	
	
	transform!(control_data,
		:trialphase => ByRow(x -> ifelse(x ∈ ["control_explore", "control_predict_homebase", "control_reward"], 1, 0)) => :trial_ptype)
	# sort!(control_data, [:record_id, :trial_index])
	transform!(groupby(control_data, :record_id),
		:trial_ptype => cumsum => :trial
	)

	control_task_data = filter(row -> row.trialphase ∈ ["control_explore", "control_predict_homebase", "control_reward"], control_data)
	control_task_data = control_task_data[:, .!all.(ismissing, eachcol(control_task_data))]
	
	control_feedback_data = filter(row -> row.trialphase ∈ ["control_explore_feedback", "control_reward_feedback"], control_data)
	control_feedback_data = control_feedback_data[:, .!all.(ismissing, eachcol(control_feedback_data))]
	
	control_report_data = filter(row -> row.trialphase ∈ ["control_confidence", "control_controllability"], control_data)
	control_report_data = control_report_data[:, .!all.(ismissing, eachcol(control_report_data))]
	select!(control_report_data, [:exp_start_time, :prolific_pid, :record_id, :session, :task, :time_elapsed, :trialphase, :trial, :rt, :response])

	extract_timeline_variables!(control_task_data)
	transform!(control_task_data, :responseTime => (x -> passmissing(JSON.parse).(x)) => :response_times)
	select!(control_task_data, Not(:responseTime))

	control_task_data = merge_control_task_and_feedback(control_task_data, control_feedback_data)

	# List of participant IDs that need swapping
	target_pids = ["67d3168115f1a0d46c1acf09", "67c6c1b45188852598a474bf", "6793624a4b2a8f2c0c31ae45"]
    
	# Find rows where prolific_pid matches any target ID
	for row in eachrow(control_task_data)
			if row.prolific_pid in target_pids
					# Temporarily store one value
					temp = row.response
					# Swap values
					row.response = row.button
					row.button = temp
			end

			if row.trialphase == "control_reward" && !ismissing(row.response)
				row.ship_color = row.response == "left" ? row.left : row.right
			end
	end

	return control_task_data, control_report_data
end

"""
    prepare_post_PILT_test_data(data::AbstractDataFrame) -> DataFrame

Processes and prepares data from the PILT test phase for further analysis. This function filters rows to include only those from the PILT test phase, removes columns where all values are missing, and computes additional columns based on participant responses.
This is for pilots prior to Pilot9 - deprecated.

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
	@assert Set(test_data.response) ⊆ Set(["ArrowRight", "ArrowLeft", "null", "right", "left", "noresp", nothing]) "Unexpected responses in PILT test data"
	
	test_data.chosen_stimulus = ifelse.(
		test_data.response .∈ (["ArrowRight", "right"],),
		test_data.stimulus_right,
		ifelse.(
			test_data.response .∈ (["ArrowLeft", "left"],),
			test_data.stimulus_left,
			missing
		)
	)

	test_data.right_chosen = (x -> get(Dict("ArrowRight" => true, "right" => true, "ArrowLeft" => false, "left" => false, "null" => missing, "noresp" => missing), x, missing)).(test_data.response)

	return test_data

end