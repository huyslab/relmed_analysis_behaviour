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
	required_columns = [participant_id_column, :version, :module_start_time, :session, :trialphase, :trial_number, :avgSpeed, :responseTime, :trialPresses]

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
			:version,
			:module_start_time,
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

function prepare_vigour_data(df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
    # Define required columns for vigour data
	required_columns = [participant_id_column, :version, :module_start_time, :session, :trialphase, :trial_number, :trial_duration, :response_time, :timeline_variables]
	required_columns = vcat(required_columns, names(df, r"(total|trial)_(reward|presses)$"))

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(df))
            insertcols!(df, col => missing)
        end
    end

	# Prepare vigour data
	vigour_data = df |>
		x -> select(x, Cols(intersect(names(df), string.(required_columns)))) |>
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
		# vigour_data = exclude_double_takers(vigour_data)
	return vigour_data
end

function prepare_PIT_data(df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )
    # Define required columns for PIT data
    required_columns = [participant_id_column, :version, :module_start_time, :session, :trialphase, :pit_trial_number, :trial_duration, :response_time, :pit_coin, :timeline_variables]
    required_columns = vcat(required_columns, names(df, r"(total|trial)_(reward|presses)$"))

    # Check and add missing columns
    for col in required_columns
        if !(string(col) in names(df))
            insertcols!(df, col => missing)
        end
    end

    # Prepare PIT data
    pit_data = df |>
        x -> select(x, Cols(intersect(names(df), string.(required_columns)))) |>
        x -> rename(x, [:pit_trial_number => :trial_number, :pit_coin => :coin]) |>
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
		# pit_data = exclude_double_takers(pit_data)
    return pit_data
end

function prepare_control_data(df::DataFrame;
    participant_id_column::Symbol = :participant_id
    )

    function extract_timeline_variables!(df::DataFrame)
        parsed = map(row -> begin
                ismissing(row.timeline_variables) && return Dict()
                str = startswith(row.timeline_variables, "{") ? row.timeline_variables : "{" * row.timeline_variables
                try
                    JSON.parse(str)
                catch e
                    @warn "Failed to parse JSON in timeline_variables" exception=(e, catch_backtrace()) value=str
                    Dict()
                end
            end, eachrow(df))
        
        for key in unique(Iterators.flatten(keys.(parsed)))
                df[!, key] = [get(p, key, missing) for p in parsed]
        end

        select!(df, Not(:timeline_variables))
        
        return df
    end

    control_data = filter(x -> !ismissing(x.trialphase) && contains(x.trialphase, r"control_.*"), df)
	# control_data = exclude_double_takers(control_data)
	
	for col in names(control_data)
		control_data[!, col] = [val === nothing ? missing : val for val in control_data[!, col]]
	end
	control_data = control_data[:, .!all.(ismissing, eachcol(control_data))]
	
    # Filter out unnecessary columns: _n_warnings, and n_instruction_fail
    select!(control_data, Not(Cols(endswith("_n_warnings"), "n_instruction_fail")))

	DataFrames.transform!(control_data,
		:trialphase => ByRow(x -> ifelse(x ∈ ["control_explore", "control_predict_homebase", "control_reward"], 1, 0)) => :trial_ptype)
	# sort!(control_data, [participant_id_column, :trial_index])
	DataFrames.transform!(groupby(control_data, [participant_id_column, :session]),
		:trial_ptype => cumsum => :trial
	)
	select!(control_data, Not(Cols(:n_warnings, :plugin_version, :pre_kick_out_warned, :trial_type, :trial_ptype)))

	control_task_data = filter(row -> row.trialphase ∈ ["control_explore", "control_predict_homebase", "control_reward"], control_data)
	control_task_data = control_task_data[:, .!all.(ismissing, eachcol(control_task_data))]
	
	control_feedback_data = filter(row -> row.trialphase ∈ ["control_explore_feedback", "control_reward_feedback"], control_data)
	control_feedback_data = control_feedback_data[:, .!all.(ismissing, eachcol(control_feedback_data))]

	sort!(control_task_data, [:module_start_time, participant_id_column, :session, :task, :version, :trial])
	sort!(control_feedback_data, [:module_start_time, participant_id_column, :session, :task, :version, :trial])
	merged_control = outerjoin(control_task_data, control_feedback_data, on=[:module_start_time, participant_id_column, :session, :task, :version, :trial], source=:source, makeunique=true, order=:left)
	if "correct_1" in names(merged_control)
		transform!(merged_control, [:correct, :correct_1] => ((x, y) -> coalesce.(x, y)) => :correct)
	end
	select!(merged_control, Not(Cols(r".*_1", :source)))

	extract_timeline_variables!(merged_control)
	transform!(merged_control, :responseTime => (x -> passmissing(JSON.parse).(x)) => :response_times)
	select!(merged_control, Not(:responseTime))
	
	control_report_data = filter(row -> row.trialphase ∈ ["control_confidence", "control_controllability"], control_data)
	control_report_data = control_report_data[:, .!all.(ismissing, eachcol(control_report_data))]
	select!(control_report_data, [:module_start_time, participant_id_column, :session, :version, :task, :time_elapsed, :trialphase, :trial, :rt, :response])

	return (; control_task = merged_control, control_report = control_report_data)
end


TASK_PREPROC_FUNCS = Dict(
    "PILT" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt", kwargs...),
    "PILT_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt_test", kwargs...),
    "WM" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm", kwargs...),
    "WM_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm_test", kwargs...),
    "reversal" => prepare_reversal_data,
    "delay_discounting" => prepare_delay_discounting_data,
    "max_press" => prepare_max_press_data,
    "vigour" => prepare_vigour_data,
    "PIT" => prepare_PIT_data,
    "control" => prepare_control_data
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


