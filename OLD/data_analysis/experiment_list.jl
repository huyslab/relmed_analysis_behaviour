
using HTTP
using JSON

const PROLIFIC_API_KEY = ENV["PROLIFIC_API_KEY"]
const BASE_URL = "https://api.prolific.com/api/v1"

function get_prolific(endpoint::String)
    headers = [
        "Authorization" => "Token $PROLIFIC_API_KEY",
        "Content-Type" => "application/json"
    ]
    response = HTTP.get("$BASE_URL$endpoint", headers)
    return JSON.parse(String(response.body))
end

function get_completed_studies()
    studies_data = get_prolific("/studies/?state=COMPLETED")
    return studies_data["results"]
end

function get_study_details(study_id::String)
    return get_prolific("/studies/$study_id/")
end

function get_approved_counts_per_study()
    completed_studies = get_completed_studies()
    approved_counts = Dict{String, Dict{String, Any}}()

    for study in completed_studies
        study_id = study["id"]
        name = study["name"]

        # Fetch full study details to get internal_name
        study_details = get_study_details(study_id)
        internal_name = get(study_details, "internal_name", "No internal name")

        submissions_data = get_prolific("/submissions/?study=$study_id")
        submissions = submissions_data["results"]
        approved_count = count(sub -> sub["status"] == "APPROVED", submissions)

        approved_counts[study_id] = Dict(
            "name" => name,
            "internal_name" => internal_name,
            "approved_count" => approved_count
        )
    end

    return approved_counts
end

# Run it
approved_counts = get_approved_counts_per_study()
for (study_id, info) in approved_counts
    println("Study ID: $study_id | Internal name: $(info["internal_name"]) => Approved submissions: $(info["approved_count"])")
end

const BASE_URL = "https://api.prolific.com/api/v1"

function get_prolific(endpoint::String)
    headers = [
        "Authorization" => "Token $PROLIFIC_API_KEY",
        "Content-Type" => "application/json"
    ]
    response = HTTP.get("$BASE_URL$endpoint", headers)
    return JSON.parse(String(response.body))
end

function get_completed_studies()
    studies_data = get_prolific("/studies/?state=COMPLETED")
    return studies_data["results"]
end

function get_approved_counts_per_study()
    completed_studies = get_completed_studies()
    approved_counts = Dict{String, Dict{String, Any}}()

    for study in completed_studies
        study_id = study["id"]
        name = study["name"]

        # Fetch full study details to get internal_name
        study_details = get_study_details(study_id)
        internal_name = get(study_details, "internal_name", "No internal name")

        submissions_data = get_prolific("/submissions/?study=$study_id")
        submissions = submissions_data["results"]
        approved_count = count(sub -> sub["status"] == "APPROVED", submissions)

        approved_counts[study_id] = Dict(
            "name" => name,
            "internal_name" => internal_name,
            "approved_count" => approved_count
        )
    end

    return approved_counts
end

# Run it
approved_counts = get_approved_counts_per_study()

# Sort by internal_name
sorted = sort(collect(approved_counts), by = x -> x[2]["internal_name"])

# Print
for (study_id, info) in sorted
    println("Study ID: $study_id | Internal name: $(info["internal_name"]) => Approved submissions: $(info["approved_count"])")
end