# Preprocess data from REDCap, dividing into tasks and preparing various variables
using DataFrames, JLD2, JSON
include("$(pwd())/core/fetch_redcap.jl")
include("$(pwd())/core/experiment-registry.jl")

# Remove rows where participant_id matches known test/demo patterns or is too short
function remove_testing!(data::DataFrame; participant_id_column::Symbol = :participant_id)
    # Exclude participant IDs matching test/demo patterns
    filter!(x -> !occursin(r"haoyang|yaniv|tore|demo|simulate|debug|REL-LON-000", x[participant_id_column]), data)
    # Exclude participant IDs with length <= 10
    filter!(x -> length(x[participant_id_column]) > 10, data)
    return data
end

remove_empty_columns(data::DataFrame) = data[:, Not(map(col -> all(ismissing, col), eachcol(data)))]

function prepare_card_choosing_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id,
    task_name::String = "pilt",
    filter_func::Function = (x -> !ismissing(x.trialphase) && x.trialphase == task_name),
    )

	# Select rows
	task_data = filter(filter_func, df)

	# Select columns
	task_data = remove_empty_columns(task_data)

	# Filter practice
	filter!(x -> isa(x.block, Int64), task_data)

	# Sort
	sort!(task_data, [participant_id_column, :session, :block, :trial])

	return task_data

end

function prepare_reversal_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
	reversal_data = filter(x -> x.trial_type == "reversal", df)

	# Select columns
	reversal_data = remove_empty_columns(reversal_data)

	# Sort
	sort!(reversal_data, [participant_id_column, :session, :block, :trial])

	return reversal_data
end

function prepare_delay_discounting_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
    delay_discounting_data = filter(x -> !ismissing(x.trialphase) && x.trialphase == "dd_task", df)

    # Select columns
    delay_discounting_data = remove_empty_columns(delay_discounting_data)

    # Sort
    sort!(delay_discounting_data, [participant_id_column, :session, :trial_index])

    return delay_discounting_data
end

function prepare_max_press_data(
    df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
	# Define required columns for max press data
	required_columns = [participant_id_column, :record_id, :version, :exp_start_time, :session, :trialphase, :trial_number, :avgSpeed, :responseTime, :trialPresses]

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(df))
            insertcols!(df, col => missing)
        end
    end

	# Prepare vigour data
	max_press_data = df |>
		x -> filter(x -> !ismissing(x.trialphase) && x.trialphase == "max_press_rate", x) |>
		x -> select(x, 
			participant_id_column,
			:record_id,
			:version,
			:exp_start_time,
			:session,
			:trialphase,
			:trial_number,
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
		# max_press_data = exclude_double_takers(max_press_data)
	return max_press_data
end


TASK_PREPROC_FUNCS = Dict(
    "PILT" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt", kwargs...),
    "PILT_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt_test", kwargs...),
    "WM" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm", kwargs...),
    "WM_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm_test", kwargs...),
    "reversal" => prepare_reversal_data,
    "delay_discounting" => prepare_delay_discounting_data,
    # "vigour" => prepare_vigour_data,
    # "PIT" => prepare_PIT_data,
    "max_press" => prepare_max_press_data,
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

    # Split and preprocess data by task
    task_data = []
    task_names = Symbol[]
    for task in experiment.tasks_included
        if haskey(TASK_PREPROC_FUNCS, task)
            @info "Preprocessing task: $task"
            task_df = TASK_PREPROC_FUNCS[task](jspsych_data; participant_id_column = experiment.participant_id_column)
            push!(task_data, task_df)
            push!(task_names, Symbol(task))
        else
            @warn "No preprocessing function defined for task: $task"
        end
    end

    task_dfs = NamedTuple{Tuple(task_names)}(task_data)
    return task_dfs
end


