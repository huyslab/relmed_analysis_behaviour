# Functions to interface with OSF

"""
    setup_osf(project_title::String) -> OSF.Project

Initializes an OSF project object for the project with the given title.

# Arguments
- `project_title::String`: The title of the OSF project.

# Returns
- An `OSF.Project` object representing the project.

# Notes
This function requires a valid OSF token stored in the `osf_token` environment variable (`ENV["osf_token"]`).
"""
function setup_osf(project_title::String)
	return OSF.project(
		OSF.Client(;
			token=ENV["osf_token"]
		); 
		title = project_title)
end


function get_file_hash(file_path::String)
    return bytes2hex(sha256(read(file_path)))
end

"""
    upload_to_osf(filepath::String, osf_project::OSF.Project, osf_folder::String; force::Bool = true)

Uploads a file to the specified folder in an Open Science Framework (OSF) project.

# Arguments
- `filepath::String`: The local path to the file you wish to upload.
- `osf_project::OSF.Project`: The OSF project object where the file will be uploaded.
- `osf_folder::String`: The folder within the OSF project where the file should be placed.
- `force::Bool`: Optional. If `true` (default), the file will be uploaded even if it exists, replacing the existing file.

# Returns
- Logs the syntax to add the image to the wiki.

# Notes
The function requires exactly one view-only link on the project to proceed with the upload.
"""
function upload_to_osf(
	filepath::String,
	osf_project::OSF.Project,
	osf_folder::String;
	force::Bool = true
)

	# Extract filename
	filename = basename(filepath)

    hash_path = filepath * ".hash"
    
    # Compute current hash of file
    current_hash = get_file_hash(filepath)
    
    # Load stored hash if it exists
    if isfile(hash_path) && !force
        stored_hash = read(hash_path, String)
        if stored_hash == current_hash
            println("File hasn't changed; skipping upload.")
            return
        end
    end
    
    # Upload file and update stored hash
    write(hash_path, current_hash)

	# Check that there is a view only link
	# @assert length(OSF.view_only_links(osf_project)) == 1 "Project needs exactly 1 view-only link for this to work"

	# OSF file object
	osf_file = OSF.file(OSF.directory(osf_project, osf_folder), filename)

	# Upload or update
	resp = cp(
		filepath, 
		osf_file, 
		force = true
	)

	url = JSON.parse(String(resp.body))["data"]["links"]["download"]

	@info """For wiki: ![image]($url =75%x)"""

end