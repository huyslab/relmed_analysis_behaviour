# Preprocess data from REDCap, dividing into tasks and preparing various variables
# Version: 1.0.3
# Last Modified: 2025-11-05
using DataFrames, JLD2, JSON
include("$(pwd())/core/fetch_redcap.jl")
include("$(pwd())/core/experiment-registry.jl")

remove_empty_columns(data::DataFrame) = data[:, Not(map(col -> all(ismissing, col), eachcol(data)))]

function exclude_retakes(
    df_original::AbstractDataFrame;
    experiment::ExperimentInfo = NORMING,
)   

    df = copy(df_original)

    df.module_start_time = DateTime.(df.module_start_time, "yyyy-mm-dd_HH:MM:SS")

    participant_id_column = experiment.participant_id_column
    module_column = experiment.module_column

    sittings = unique(df[!, [participant_id_column, :session, module_column, :module_start_time]])

    transform!(
        groupby(sittings, [participant_id_column, :session, module_column]),
        :module_start_time => minimum => :first_start_time
    )

    pre = nrow(sittings)

    filter!(x -> x.module_start_time == x.first_start_time, sittings)

    @info "Excluded $(pre - nrow(sittings)) retake module attempts (kept the first attempt)"

    df = innerjoin(df, select(sittings, Not(:first_start_time)), on=[participant_id_column, :session, module_column, :module_start_time])

    return df
end

function prepare_card_choosing_data(
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    task_name::String = "pilt",
    filter_func::Function = (x -> !ismissing(x.trialphase) && x.trialphase == task_name),
    )

    participant_id_column = experiment.participant_id_column

	# Select rows
	task_data = filter(filter_func, df)

	# Select columns
	task_data = remove_empty_columns(task_data)

	# Filter practice but not PIT test data
	task_name == "pit_test" ? task_data : filter!(x -> isa(x.block, Int64), task_data)

	# Sort
	sort!(task_data, [participant_id_column, :session, :block, :trial])

    if task_name == "pilt"
        stimuli_feedback = vcat(
            unique(select(task_data, :session, :block, :trial, :feedback_right => :feedback, :stimulus_right => :stimulus)),
            unique(select(task_data, :session, :block, :trial, :feedback_left => :feedback, :stimulus_left => :stimulus))
        )

        # Remove prefix from stimulus columns
        transform!(
            stimuli_feedback,
            :stimulus => ByRow(x -> ismissing(x) ? x : replace(x, r"^\./assets/images/card-choosing/stimuli/" => "")) => :stimulus
        )

        sort!(stimuli_feedback, [:session, :block, :trial, :stimulus])

        common_feedback = combine(
            groupby(stimuli_feedback, [:session, :block, :stimulus]),
            :feedback => mode => :common_feedback,
            :feedback => (x -> begin
                rare = unique(filter(y -> y .!= mode(x), x))
                isempty(rare) ? "None" : only(rare)
            end) => :rare_feedback,
            :feedback => (x -> length(unique(x))) => :n_feedback_types
        )

        @assert all(common_feedback.n_feedback_types .<= 2) "Too many feedback values found for some stimuli."

        valence_perblock = combine(
            groupby(common_feedback, [:session, :block]),
            :common_feedback => (x -> ifelse(all(x .> 0), "Reward", ifelse(all(x .< 0), "Punishment", "Mixed"))) => :valence
        )
        
        leftjoin!(select!(task_data, Not(:valence)), valence_perblock, on=[:session, :block])
    end
	
    return task_data

end


function prepare_reversal_data(
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)

    participant_id_column = experiment.participant_id_column


	reversal_data = filter(x -> x.trial_type == "reversal", df)

	# Select columns
	reversal_data = remove_empty_columns(reversal_data)

	# Sort
	sort!(reversal_data, [participant_id_column, :session, :block, :trial])

	return reversal_data
end

function prepare_delay_discounting_data(
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)

    participant_id_column = experiment.participant_id_column


    delay_discounting_data = filter(x -> !ismissing(x.trialphase) && x.trialphase == "dd_task", df)

    # Select columns
    delay_discounting_data = remove_empty_columns(delay_discounting_data)

    # Sort
    sort!(delay_discounting_data, [participant_id_column, :session, :trial_index])

    return delay_discounting_data
end

function prepare_max_press_data(
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)

    participant_id_column = experiment.participant_id_column


	# Define required columns for max press data
	required_columns = [participant_id_column, :module_start_time, :session, :trialphase, :trial_number, :avgSpeed, :responseTime, :trialPresses]

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
			:session,
            :module_start_time,
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
            :responseTime => ByRow(x -> ismissing(x) ? missing : JSON.parse(x)) => :response_times
		) |>
		x -> select(x, 
			Not([:responseTime])
		)
		# max_press_data = exclude_double_takers(max_press_data)
	return max_press_data
end

function prepare_piggybank_data(df::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    task::String = "vigour"  # "vigour" or "PIT"
)
    participant_id_column = experiment.participant_id_column

    # Define base and type-specific columns
    base_columns = [participant_id_column, :module_start_time, :session, :trialphase, :trial_duration, :response_time, :timeline_variables]

    if task == "vigour"
        specific_columns = [:trial_number]
        renames = []
        filter_condition = [:trialphase, :trial_number] => ByRow((x, y) -> (!ismissing(x) && x in ["vigour_trial"]) && (!ismissing(y)))
    elseif task == "PIT"
        specific_columns = [:pit_trial_number, :pit_coin]
        renames = [:pit_trial_number => :trial_number, :pit_coin => :coin]
        filter_condition = :trialphase => ByRow(x -> !ismissing(x) && x in ["pit_trial"])
    else
        @error "Unknown task type: $task"
    end

    required_columns = vcat(base_columns, specific_columns, names(df, r"(total|trial)_(reward|presses)$"))

    # Add missing columns
    for col in required_columns
        if !(string(col) in names(df))
            insertcols!(df, col => missing)
        end
    end

    # Process data
    result = df |>
        x -> subset(x, filter_condition) |>
        x -> select(x, Cols(intersect(names(df), string.(required_columns)))) |>
        x -> ((df) -> isempty(renames) ? df : rename(df, renames...))(x) |>
        x -> DataFrames.transform(x,
            :response_time => ByRow(x -> ismissing(x) ? missing : JSON.parse(x)) => :response_times,
            :timeline_variables => ByRow(x -> JSON.parse(x)["ratio"]) => :ratio,
            :timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]) => :magnitude,
            :timeline_variables => ByRow(x -> JSON.parse(x)["magnitude"]/JSON.parse(x)["ratio"]) => :reward_per_press
        ) |>
        x -> select(x, Not([:response_time, :timeline_variables]))

    return result
end

# Wrapper functions for backward compatibility
prepare_vigour_data(df::DataFrame; experiment::ExperimentInfo = TRIAL1) =
    prepare_piggybank_data(df; experiment = experiment, task = "vigour")

prepare_PIT_data(df::DataFrame; experiment::ExperimentInfo = TRIAL1) =
    prepare_piggybank_data(df; experiment = experiment, task = "PIT")

function prepare_vigour_test_data(
    df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)

    participant_id_column = experiment.participant_id_column

    # Define required columns for vigour test data
	required_columns = [participant_id_column, :module_start_time, :session, :trialphase, :response, :rt]
    renames  = [:rt => :response_times]
	required_columns = vcat(required_columns, names(df, r"(magnitude|ratio)$"))

	# Check and add missing columns
	for col in required_columns
        if !(string(col) in names(df))
            insertcols!(df, col => missing)
        end
    end

	# Process post vigour test data
	result = subset(df, :trialphase => ByRow(x -> !ismissing(x) && x in ["vigour_test"])) |>
		x -> select(x, Cols(intersect(names(df), string.(required_columns)))) |>
        x -> ((df) -> isempty(renames) ? df : rename(df, renames...))(x)

    return result
end

function prepare_control_data(df::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    )

    participant_id_column = experiment.participant_id_column
    module_column = experiment.module_column

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
            if (key in names(df))
                @warn "Column $key already exists in DataFrame. Overwriting with parsed timeline variable."
            end
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
    select!(control_data, Not(Cols(intersect(names(control_data), [names(control_data, endswith("_n_warnings"))..., "n_instruction_fail"]))))

    select!(control_data, Not(Cols(intersect(names(control_data), ["n_warnings", "plugin_version", "pre_kick_out_warned"]))))

    extract_timeline_variables!(control_data)
	transform!(control_data, :responseTime => (x -> passmissing(JSON.parse).(x)) => :response_times)
	select!(control_data, Not(:responseTime))

    # Forward fill trial numbers
    ffill(v) = v[accumulate(max, [i*!ismissing(v[i]) for i in 1:length(v)], init=1)]

    filter!(row -> row.trialphase ∈ ["control_explore", "control_predict_homebase", "control_reward", "control_explore_feedback", "control_confidence", "control_controllability"], control_data)
    sort!(control_data, [participant_id_column, :session, :module_start_time, :trial_index])
    transform!(control_data, :trial => ffill => :trial)

	control_task_data = filter(row -> row.trialphase ∈ ["control_explore", "control_predict_homebase", "control_reward"], control_data)
	control_task_data = control_task_data[:, .!all.(ismissing, eachcol(control_task_data))]
	
	control_feedback_data = filter(row -> row.trialphase ∈ ["control_explore_feedback", "control_reward_feedback"], control_data)
	control_feedback_data = control_feedback_data[:, .!all.(ismissing, eachcol(control_feedback_data))]

	sort!(control_task_data, [:module_start_time, participant_id_column, :session, module_column, :trial])
	sort!(control_feedback_data, [:module_start_time, participant_id_column, :session, module_column, :trial])
	merged_control = outerjoin(control_task_data, control_feedback_data, on=[:module_start_time, participant_id_column, :session, module_column, :trial], source=:source, makeunique=true, order=:left)
	if "correct_1" in names(merged_control)
		transform!(merged_control, [:correct, :correct_1] => ((x, y) -> coalesce.(x, y)) => :correct)
	end
	select!(merged_control, Not(Cols(r".*_1", :source)))

    # Warn if trials exceed expected number; maybe suggest double-takers
    for group in groupby(merged_control, [participant_id_column, :session])
        n_trials = length(group.trial)
        expected_trials = group.session[1] == "screening" ? 28 : 120
        if n_trials != expected_trials
            @warn "$(group[1, participant_id_column]) in $(group[1, :session]) has control trials: $(n_trials) (expected $expected_trials)"
        end
    end
	
	control_report_data = filter(row -> row.trialphase ∈ ["control_confidence", "control_controllability"], control_data)
	control_report_data = control_report_data[:, .!all.(ismissing, eachcol(control_report_data))]

    if !isempty(control_report_data)
	    select!(control_report_data, [:module_start_time, participant_id_column, :session, module_column, :time_elapsed, :trialphase, :trial, :rt, :response])
    end

	return (; control_task = merged_control, control_report = control_report_data)
end

function prepare_questionnaire_data(
    df::AbstractDataFrame;
    experiment::ExperimentInfo = TRIAL1
    )

    participant_id_column = experiment.participant_id_column

	raw_questionnaire_data = filter(x -> !ismissing(x.trialphase) && 
        x.trialphase in experiment.questionnaire_names, df)

    function merge_keys!(dict, prefix)
        keys_to_merge = filter(k -> startswith(k, prefix), keys(dict))
        values = [dict[k] for k in keys_to_merge if haskey(dict, k)]
        non_empty_values = filter(x -> !ismissing(x) && x != "", values)
        for k in keys_to_merge
            delete!(dict, k)
        end
        dict[prefix] = isempty(non_empty_values) ? "" : join(non_empty_values, "; ")
        return dict
    end

	questionnaire_data = DataFrame()
	for row in eachrow(raw_questionnaire_data)
		response = nothing
		try
			response = row.trial_type in ["survey-template", "survey-demo"] ? JSON.parse(row.responses) : JSON.parse(row.response)

            # Special handling for demographics questionnaire to merge choice/text fields
            if row.trialphase == "demographics"
                for prefix in ["menstrual-first-day", "menstrual-cycle-length"]
                    merge_keys!(response, prefix)
                end
                if haskey(response, "gender-free-response")
                    response["gender-other"] = pop!(response, "gender-free-response")
                end
			end
		catch e
			@warn "Failed to parse JSON in questionnaire data" participant_id=row[participant_id_column] trialphase=row.trialphase error=e
			continue
		end
		for (key, value) in response
			push!(questionnaire_data,
				NamedTuple{(
					participant_id_column,
					:module_start_time,
					:session,
					experiment.module_column,
					:trialphase,
                    :trial_type,
					:question,
					:response
				)}((
					row[participant_id_column],
					row.module_start_time,
					row.session,
                    row[experiment.module_column],
					row.trialphase,
                    row.trial_type,
					key,
					value
				)); promote=true)
		end
	end

	# Add question_id
	insertcols!(
		questionnaire_data,
		5,
		:question_id => ((t, q) -> "$(t)_$q").(questionnaire_data.trialphase, questionnaire_data.question)
	)

	return questionnaire_data
end

function prepare_pavlovian_lottery_data(df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
) 

    participant_id_column = experiment.participant_id_column

    pavlovian_lottery_data = filter(x -> !ismissing(x.trialphase) && x.trialphase == "prepilt_conditioning", df)

    # Select columns
    pavlovian_lottery_data = remove_empty_columns(pavlovian_lottery_data)

    # Parse timeline variables
    transform!(
        pavlovian_lottery_data,
        :timeline_variables => ByRow(x -> JSON.parse(x)["pav_value"]) => :pavlovian_value,
        :timeline_variables => ByRow(x -> JSON.parse(x)["prepilt_trial"]) => :trial
    )
    select!(pavlovian_lottery_data, Not(:timeline_variables))

    # Sort
    sort!(pavlovian_lottery_data, [participant_id_column, :session, :trial])

    return pavlovian_lottery_data

end

function prepare_open_text_data(df::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)

    participant_id_column = experiment.participant_id_column

    open_text_data = filter(x -> !ismissing(x.trialphase) && x.trialphase == "open-text", df)

    # Select columns
    open_text_data = remove_empty_columns(open_text_data)

    transform!(
        open_text_data,
        :response => ByRow(x -> only(keys(JSON.parse(x)))) => :question,
        :response => ByRow(x -> only(values(JSON.parse(x)))) => :response
    )

    # Sort
    sort!(open_text_data, [participant_id_column, :session, :question])

    return open_text_data

end


TASK_PREPROC_FUNCS = Dict(
    "PILT" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt", kwargs...),
    "PILT_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pilt_test", kwargs...),
    "WM" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm", kwargs...),
    "WM_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "wm_test", kwargs...),
    "reversal" => prepare_reversal_data,
    "delay_discounting" => prepare_delay_discounting_data,
    "vigour" => prepare_vigour_data,
    "vigour_test" => prepare_vigour_test_data,
    "PIT" => prepare_PIT_data,
    "PIT_test" => (x; kwargs...) -> prepare_card_choosing_data(x; task_name = "pit_test", filter_func = (x -> !ismissing(x.trialphase) && x.trialphase == "pilt_test" && x.block == "pavlovian"), kwargs...),
    "max_press" => prepare_max_press_data,
    "control" => prepare_control_data,
    "questionnaire" => prepare_questionnaire_data,
    "pavlovian_lottery" => prepare_pavlovian_lottery_data,
    "open_text" => prepare_open_text_data
)

function preprocess_project(
    experiment::ExperimentInfo;
    force_download::Bool = false,
    delay_ms::Int = 100,
    use_manual_download::Bool = false
)

    # Create data folder if doesn't exist
    isdir("data") || mkpath("data")

	datafile = "data/$(experiment.project)/$(experiment.project).jld2"

	# Load data or download from REDCap
	if !isfile(datafile) || force_download
        # Fetch data from REDCap or local files
        if use_manual_download
            local_data_dir = "data/$(experiment.project)/manual_download"
            @info "Loading data from local files in: $local_data_dir"
            jspsych_data = get_local_files(local_data_dir)
        else
            @info "Fetching data from REDCap API"
            jspsych_data = fetch_project_data(project = experiment.project; delay_ms = delay_ms)
        end
        
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
    jspsych_data = experiment.exclude_testing_participants(jspsych_data; experiment = experiment)

    # Remove retakes if specified
    if experiment.exclude_retakes
        jspsych_data = exclude_retakes(jspsych_data; experiment = experiment)
    end

    # Split and preprocess data by task
    task_data = []
    task_names = Symbol[]
    for task in experiment.tasks_included
        if haskey(TASK_PREPROC_FUNCS, task)
            @info "Preprocessing task: $task"
            task_df = TASK_PREPROC_FUNCS[task](jspsych_data; experiment = experiment)
            push!(task_data, task_df)
            push!(task_names, Symbol(task))
        else
            @warn "No preprocessing function defined for task: $task"
        end
    end

    push!(task_names, :jspsych_data)
    push!(task_data, jspsych_data)

    task_dfs = NamedTuple{Tuple(task_names)}(task_data)
    return task_dfs
end


