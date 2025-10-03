using DataFrames, JSON
function load_PILT_sequence(
    filename::String
)
    # Read the JavaScript file
    js_content = read(filename, String)
    
    # Find the PILT_json declaration using regex
    pattern = r"const\s+PILT_json\s*=\s*'([^']*)';"
    match_result = match(pattern, js_content)
    
    if match_result === nothing
        error("Could not find PILT_json declaration in file: $filename")
    end
    
    # Extract the value (first capture group)
    pilt_json_value = match_result.captures[1]
    
    # Parse the JSON string
    parsed_data = JSON.parse(pilt_json_value)
    
    # Flatten the array of arrays of dicts into a single array
    flattened_data = []
    for block_array in parsed_data
        for trial_dict in block_array
            push!(flattened_data, trial_dict)
        end
    end
    
    # Create DataFrame with only specified columns
    df = DataFrame(
        block = [trial["block"] for trial in flattened_data],
        trial = [trial["trial"] for trial in flattened_data],
        feedback_common = [trial["feedback_common"] for trial in flattened_data],
        feedback_left = [trial["feedback_left"] for trial in flattened_data],
        feedback_right = [trial["feedback_right"] for trial in flattened_data]
    )
    
    return df
end