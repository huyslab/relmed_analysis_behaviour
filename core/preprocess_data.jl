# Preprocess data from REDCap, dividing into tasks and preparing various variables

remove_testing!(data::DataFrame; participant_id_field::Symbol = :participant_id) = filter!(x -> (!occursin(r"haoyang|yaniv|tore|demo|simulate|debug", x[participant_id_field])) && (length(x[participant_id_field]) > 10), data)

remove_empty_columns(data::DataFrame) = data[:, Not(map(col -> all(ismissing, col), eachcol(data)))]

function prepare_PILT_data(
    df::DataFrame;
    participant_id_field::Symbol = :participant_id,
    filter_func::Function = (x -> !ismissing(x.trialphase) && x.trialphase == "pilt"),
    )

	# Select rows
	PILT_data = filter(filter_func, df)

	# Select columns
	PILT_data = remove_empty_columns(PILT_data)

	# Filter practice
	filter!(x -> typeof(x.block) == Int64, PILT_data)

	# Sort
	sort!(PILT_data, [participant_id_field, :session, :block, :trial])

	return PILT_data

end


const TASK_PREPROC_FUNCS = Dict(
    "PILT" => prepare_PILT_data,
    # "post_PILT_test" => prepare_test_data,
    # "vigour" => prepare_vigour_data,
    # "PIT" => prepare_PIT_data,
    # "max_press" => prepare_max_press_data,
    # "control" => prepare_control_data
)

function preprocess_project(
    experiment::ExperimentInfo;
    force_download::Bool = false
)

    # Create data folder if doesn't exist
    isdir("data") || mkpath("data")

	datafile = "data/$(experiment.project).jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
        # Fetch data from REDCap
        jspsych_data = fetch_project_data(project = experiment.project)
        
        if jspsych_data === nothing
            @warn "No data found for project: $(experiment.project)"
            return nothing
        end

        # Save data locally
		JLD2.@save datafile jspsych_data
	else
        # Load data from local file
		JLD2.@load datafile jspsych_data
	end

    # Remove testing data
    jspsych_data = remove_testing!(jspsych_data)

    # # Split and preprocess data by task
    # task_data = []
    # task_names = Symbol[]
    # for task in experiment.tasks_included
    #     if haskey(TASK_PREPROC_FUNCS, task)
    #         @info "Preprocessing task: $task"
    #         task_df = TASK_PREPROC_FUNCS[task](jspsych_data; participant_id_field = experiment.participant_id_field)
    #         push!(task_data, task_df)
    #         push!(task_names, Symbol(task))
    #     else
    #         @warn "No preprocessing function defined for task: $task"
    #     end
    # end

    # task_dfs = NamedTuple{Tuple(task_names)}(task_data)
    # return task_dfs
end


