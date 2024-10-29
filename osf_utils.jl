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

"""
    upload_to_osf(filepath::String, osf_project::OSF.Project, osf_folder::String; force::Bool = true)

Uploads a file to the specified folder in an Open Science Framework (OSF) project.

# Arguments
- `filepath::String`: The local path to the file you wish to upload.
- `osf_project::OSF.Project`: The OSF project object where the file will be uploaded.
- `osf_folder::String`: The folder within the OSF project where the file should be placed.
- `force::Bool`: Optional. If `true` (default), the file will be uploaded even if it exists, replacing the existing file.

# Returns
- Logs the filename, folder, and OSF file ID upon successful upload or update.

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

	# Check that there is a view only link
	@assert length(OSF.view_only_links(osf_project)) == 1 "Project needs exactly 1 view-only link for this to work"

	# OSF file object
	osf_file = OSF.file(OSF.directory(osf_project, osf_folder), filename)

	# Upload or update
	resp = cp(
		filepath, 
		osf_file, 
		force = true
	)

	# Get file url
	url = OSF.url(osf_file)

	# Get file id
	file_id = split(string(url), "/")[end-1]

	@info """Saved $filename to "$osf_folder", id is $file_id"""

	return 

end