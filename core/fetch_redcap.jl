# Function to fetch data from REDCap
# Version: 1.1.1
# Last Modified: 2025-09-28
using HTTP, JSON, DataFrames

"""
    get_redcap_records(; project="trial1") -> Vector{Any}

Fetches and returns all records from the REDCap API as a JSON-parsed vector.

# Arguments
- `project::String`: The project name used to construct the token environment variable (default: "trial1")

# Returns
- `Vector{Any}`: A vector containing the parsed JSON data for all records.

# Notes
- Requires `REDCAP_TOKEN_<PROJECT>` and `REDCAP_URL` to be set in the environment variables.
- The token environment variable is constructed as `REDCAP_TOKEN_\$(uppercase(project))`.
"""
function get_redcap_records(;
		project::String = "trial1"
	)

	# Check required environment variables
	token_var = "REDCAP_TOKEN_$(uppercase(project))"
	if !haskey(ENV, token_var)
		error("Environment variable $token_var not found. Please add it to env.list")
	end
	if !haskey(ENV, "REDCAP_URL")
		error("Environment variable REDCAP_URL not found. Please add it to env.list")
	end

	# Get the records --------
	# Create the payload for getting the record details
	rec_payload = Dict(
		"token" => ENV["REDCAP_TOKEN_$(uppercase(project))"],
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
	records = HTTP.post(ENV["REDCAP_URL"], body=HTTP.Form(rec_payload), verbose = true)

	
    # Parse the JSON response and return
	return JSON.parse(String(records.body))
end

"""
    get_redcap_file(record_id; field="data", project="trial1") -> Dict

Fetches a specific file from REDCap for a given record.

# Arguments
- `record_id::String`: The ID of the record to fetch the file for
- `field::String`: The field name containing the file (default: "data")
- `project::String`: The project name (default: "trial1")

# Returns
- `Dict`: A dictionary containing "jspsych_data" and "interaction_data" keys, or empty dict if failed
"""
function get_redcap_file(
	record_id::String;
	field::String = "data",
	project::String = "trial1" 
)
	# Check required environment variables
	token_var = "REDCAP_TOKEN_$(uppercase(project))"
	if !haskey(ENV, token_var)
		error("Environment variable $token_var not found. Please add it to env.list")
	end
	if !haskey(ENV, "REDCAP_URL")
		error("Environment variable REDCAP_URL not found. Please add it to env.list")
	end

	# Create the payload for getting the file
	file_payload = Dict(
		"token" => ENV["REDCAP_TOKEN_$(uppercase(project))"],
	    "content" => "file",
		"action" => "export",
		"record" => record_id,
		"field" => field,
		"returnFormat" => "json"
	)

	# Make the POST request to the REDCap API with failsafe
	try
		file = HTTP.post(ENV["REDCAP_URL"], body=HTTP.Form(file_payload), verbose = true)
		
		# Check HTTP response status
		if file.status < 200 || file.status >= 300
			error("Failed to fetch file from REDCap API. HTTP status: $(file.status)")
		end
		# Parse
		parsed_string = JSON.parse(String(file.body))[1]

		# Check if the data contains expected keys for new format
		if !haskey(parsed_string, "jspsych_data") || !haskey(parsed_string, "interaction_data")
			@warn "Required keys 'jspsych_data' or 'interaction_data' not found for record $record_id. This is probably old format data."
            return Dict(
                "jspsych_data" => isa(parsed_string, Dict) ? [parsed_string] : [],
                "interaction_data" => []
            )
		end

		# Parse nested JSON strings
		jspsych_data = JSON.parse(parsed_string["jspsych_data"])
		interaction_data = JSON.parse(parsed_string["interaction_data"])

		# Return structured data as vector of objects
		return Dict(
			"jspsych_data" => isa(jspsych_data, Dict) ? [jspsych_data] : jspsych_data,
			"interaction_data" => isa(interaction_data, Dict) ? [interaction_data] : interaction_data
		)
	catch e
		@warn "No file could be downloaded for record_id: $record_id. Error: $e"
		return Dict()
	end
end


"""
    redcap_data_to_df(file_data::AbstractDict) -> DataFrame

Converts REDCap file data to a DataFrame format for analysis.

# Arguments
- `file_data::AbstractDict`: Dictionary containing "jspsych_data" and "interaction_data" keys

# Returns
- `DataFrame`: Combined DataFrame with jsPsych trial data and browser interaction data
"""
function redcap_data_to_df(file_data::AbstractDict)

	# Extract jsPsych and interaction data from the dictionary
	jspsych_data = file_data["jspsych_data"]
	interaction_data = file_data["interaction_data"]
	
	# Helper function to convert arrays and dicts to JSON strings for DataFrame storage
	function jsonify_multidimensional_data(d::Dict)
	    for (k, v) in d
	        if (v isa AbstractArray) || (v isa Dict)
				d[k] = json(v)
			end
	    end
	    return d
	end

	# Convert jsPsych data to DataFrame (each trial becomes a row)
	jspsych_data = vcat(
		[DataFrame(jsonify_multidimensional_data(d)) for d in jspsych_data]...,
		cols=:union
	)

    # Handle case where there's no interaction data
	if isempty(interaction_data)
		insertcols!(jspsych_data, :browser_interactions => missing, :browser_interaction_times => missing)
		return jspsych_data
	end

	# Convert interaction data to DataFrame
	interaction_data = vcat(DataFrame.(interaction_data)...)

	# Group interaction events by trial and collect into arrays
	interaction_data_colated = combine(groupby(interaction_data, :trial), 
                :event => (x -> [collect(x)]) => :browser_interactions,
                :time => (x -> [collect(x)]) => :browser_interaction_times)

	# Join the interaction data with jsPsych data on trial index
	leftjoin!(
		jspsych_data,
		rename(interaction_data_colated, :trial => :trial_index),
		on = :trial_index
	)

	return jspsych_data
end

function fetch_project_data(; 
	project::String = "trial1",
	filter_func::Function = (x -> true)
)

	# Fetch records
	records = get_redcap_records(project = project)

    # Select records
    filter!(filter_func, records)

    # Get record_id
    record_ids = [record["record_id"] for record in records]

    # Fetch the file for the given record_id. Direct to the correct REDCap project
	file_data = (x -> get_redcap_file(x; project = project)).(record_ids)

	# Filter out empty file_data
	file_data = filter(x -> !isempty(x), file_data)

	if isempty(file_data)
		return nothing
	end

	# Convert to DataFrame
    jspsych_data = redcap_data_to_df.(file_data)

    return vcat(jspsych_data..., cols=:union)

end
