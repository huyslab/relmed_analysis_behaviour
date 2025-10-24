using DataFrames, JSON

"""
    extract_json_from_js_file(filepath::String)

Read a JavaScript file and extract the JSON strings for PILT_json and reversal_json constants.

Returns a dictionary with keys "PILT_json" and "reversal_json" containing the parsed JSON data,
or missing values if the constants are not found.

# Arguments
- `filepath::String`: Path to the JavaScript file to read

# Returns
- `Dict{String, Any}`: Dictionary containing the extracted JSON data
"""
function extract_json_from_js_file(filepath::String)
    # Read the entire file content
    content = read(filepath, String)
    
    # Initialize result dictionary
    result = Dict{String, Any}()
    
    # Define patterns to search for
    patterns = [
        "PILT_json" => r"const\s+PILT_json\s*=\s*'([^']+)'",
        "reversal_json" => r"const\s+reversal_json\s*=\s*'([^']+)'"
    ]
    
    # Extract JSON strings for each pattern
    for (key, pattern) in patterns
        match_result = match(pattern, content)
        if match_result !== nothing
            json_string = match_result.captures[1]
            try
                # Parse the JSON string
                parsed_json = DataFrame(vcat(JSON.parse(json_string)...))
                result[key] = parsed_json
                println("✓ Successfully extracted and parsed $key")
            catch e
                println("⚠ Warning: Failed to parse JSON for $key: $e")
                result[key] = json_string  # Store as string if parsing fails
            end
        else
            println("⚠ Warning: $key not found in $filepath")
            result[key] = missing
        end
    end
    
    return result
end

"""
    extract_all_sequences(directory::String = "task_sequences")

Extract JSON data from all JavaScript files in the specified directory and convert to DataFrames.

# Arguments
- `directory::String`: Directory containing the JavaScript sequence files (default: "task_sequences")

# Returns
- `Tuple{DataFrame, DataFrame}`: Two DataFrames (PILT_data, reversal_data) containing all extracted data

"""
function extract_all_sequences(directory::String = "task_sequences")
    # Find all .js files in the directory and subdirectories
    js_files = [relpath(joinpath(root, f), directory) for (root, _, files) in walkdir(directory) for f in files if endswith(f, ".js")]
    
    # Initialize result dictionaries
    pilt_data = DataFrame()
    reversal_data = DataFrame()
    
    println("Found $(length(js_files)) JavaScript files:")
    
    for filename in js_files
        filepath = joinpath(directory, filename)
        println("\nProcessing: $filename")

        session = split(split(filename, "_")[2], ".")[1] # Extract session name from filename

        try
            data = extract_json_from_js_file(filepath)
            
            # Store PILT_json data
            if haskey(data, "PILT_json") && data["PILT_json"] !== missing
                pilt_data = vcat(pilt_data, data["PILT_json"])
                println("  ✓ PILT_json extracted and converted to DataFrame from file $filename ($(nrow(data["PILT_json"])) rows)")
            end
            
            # Store reversal_json data
            if haskey(data, "reversal_json") && data["reversal_json"] !== missing
                reversal_data = vcat(reversal_data, insertcols(data["reversal_json"], 1, :session => session))
                println("  ✓ reversal_json extracted and converted to DataFrame from file $filename ($(nrow(data["reversal_json"])) rows)")
            end
            
        catch e
            println("  ❌ Error processing $filename: $e")
        end
    end
    
    return pilt_data, reversal_data
end