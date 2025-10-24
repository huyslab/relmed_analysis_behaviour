using DataFrames
script_dir = dirname(@__FILE__)
include(joinpath(script_dir, "task-simulation.jl"))
include(joinpath(script_dir, "extract-task-sequences.jl"))

"""
    calculate_completion_times(jspsych_data)

Calculate task completion times in MM:SS format from jsPsych data.
Returns unstacked DataFrame with completion times per task.
"""
function calculate_completion_times(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)
    completion_times = combine(
        groupby(jspsych_data, [experiment.participant_id_column, :session, experiment.module_column]),
        :time_elapsed => (x -> begin
            total_seconds = (maximum(x) - minimum(x)) / 1000
            minutes = floor(Int, total_seconds / 60)
            seconds = floor(Int, total_seconds % 60)
            "$(minutes):$(lpad(seconds, 2, '0'))"
        end) => :completion_time,
    )

    return unstack(completion_times, [experiment.participant_id_column, :session], experiment.module_column, :completion_time;
                  renamecols = x -> Symbol("completion_time_$(x)"))
end

"""
    calculate_pilt_quiz_attempts(jspsych_data)

Count the number of attempts for PILT instruction quiz per participant/session.
"""
function calculate_pilt_quiz_attempts(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1
    )
    pilt_quiz = filter(x -> !ismissing(x.trialphase) && x.trialphase == "instruction_quiz", jspsych_data)
    return combine(
        groupby(pilt_quiz, [experiment.participant_id_column, :session, experiment.module_column]),
        :trial_index => length => :n_pilt_quiz_attempts
    )
end

"""
    process_reversal_task(jspsych_data)

Process reversal learning task data to extract missing trials and reaction times.
Returns tuple of (missing_data_df, rt_data_df).
"""
function process_reversal_task(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)
    reversal = filter(x -> x.trial_type == "reversal", jspsych_data)
    isempty(reversal) && return (DataFrame(), DataFrame())
    
    # Calculate missing responses (valid responses are "left" or "right")
    reversal.response_missing = (x -> x ∉ ["left", "right"]).(reversal.response)
    missing_data = combine(
        groupby(reversal, [experiment.participant_id_column, :session, experiment.module_column]),
        :response_missing => sum => :n_missing_trials,
        :response_missing => length => :n_trials,
        :response_missing => mean => :prop_missing_trials
    )
    missing_data = insertcols(missing_data, :trialphase => "reversal")
    
    # Calculate reaction times
    # Note: Missing RT is `nothing` instead of `missing` in the data
    rt_data = combine(
        groupby(filter(x -> !(ismissing(x.rt) || isnothing(x.rt)), reversal), [experiment.participant_id_column, :session, experiment.module_column]),
        :rt => (x -> "$(Int(round(mean(x)))) ($(Int(round(std(x)))))") => :rt
    )
    rt_data = insertcols(rt_data, :trialphase => "reversal")
    
    return (missing_data, rt_data)
end

"""
    process_pilt_task(jspsych_data)

Process PILT (Probabilistic Instrumental Learning Task) data.
Returns tuple of (missing_data_df, rt_data_df).
"""
function process_pilt_task(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
)
    pilt = filter(x -> !ismissing(x.trialphase) && x.trialphase == "pilt", jspsych_data)
    isempty(pilt) && return (DataFrame(), DataFrame())
    
    # Calculate missing responses (valid responses are "left", "middle", "right")
    pilt.response_missing = (x -> x ∉ ["left", "middle", "right"]).(pilt.response)
    missing_data = combine(
        groupby(pilt, [experiment.participant_id_column, :session, experiment.module_column, :trialphase]),
        :response_missing => sum => :n_missing_trials,
        :response_missing => length => :n_trials,
        :response_missing => mean => :prop_missing_trials
    )
    
    # Calculate reaction times
    rt_data = combine(
        groupby(filter(x -> !ismissing(x.rt), pilt), [experiment.participant_id_column, :session, experiment.module_column, :trialphase]),
        :rt => (x -> "$(Int(round(mean(x)))) ($(Int(round(std(x)))))") => :rt
    )
    
    return (missing_data, rt_data)
end

"""
    process_control_tasks(jspsych_data)

Process control task data including control_explore, control_reward, and control_predict_homebase phases.
Returns DataFrame with missing trial information.
"""
function process_control_tasks(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)
    missing_trials = DataFrame()
    
    # Control presses: require at least 1 press and valid response
    control_presses = filter(x -> !ismissing(x.trialphase) && x.trialphase ∈ ["control_explore", "control_reward"], jspsych_data)
    if !isempty(control_presses)
        # Missing if < 1 press OR invalid response
        control_presses.response_missing = (control_presses.trial_presses .< 1) .|| (x -> x ∉ ["left", "right"]).(control_presses.response)
        control_presses_missing = combine(
            groupby(control_presses, [experiment.participant_id_column, :session, experiment.module_column]),
            :response_missing => sum => :n_missing_trials,
            :response_missing => length => :n_trials,
            :response_missing => mean => :prop_missing_trials
        )
        control_presses_missing = insertcols(control_presses_missing, :trialphase => "control_presses")
        missing_trials = vcat(missing_trials, control_presses_missing)
    end
    
    # Control choice: predict homebase location
    control_choice = filter(x -> !ismissing(x.trialphase) && x.trialphase == "control_predict_homebase", jspsych_data)
    if !isempty(control_choice)
        # Missing if button response is missing
        control_choice.response_missing = ismissing.(control_choice.button)
        control_choice_missing = combine(
            groupby(control_choice, [experiment.participant_id_column, :session, experiment.module_column]),
            :response_missing => sum => :n_missing_trials,
            :response_missing => length => :n_trials,
            :response_missing => mean => :prop_missing_trials
        )
        control_choice_missing = insertcols(control_choice_missing, :trialphase => "control_choice")
        missing_trials = vcat(missing_trials, control_choice_missing)
    end
    
    return missing_trials
end

"""
    consolidate_missing_trials_and_rts(all_missing_trials, all_rts)

Transform missing trials and reaction time data into wide format.
Renames 'task' to 'module' and 'trialphase' to 'task' for consistency.
Returns tuple of (missing_trials_df, rts_df).
"""
function consolidate_missing_trials_and_rts(
    all_missing_trials::DataFrame, 
    all_rts::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)
    # Process reaction times into wide format
    rename!(all_rts, experiment.module_column => :module, :trialphase => :task)
    rts = unstack(all_rts, [experiment.participant_id_column, :session], :task, :rt; 
                 renamecols = x -> Symbol("rt_$(x)"))
    
    # Process missing trials into wide format
    rename!(all_missing_trials, experiment.module_column => :module, :trialphase => :task)
    
    # Calculate overall proportion of missing trials across all tasks
    missing_trials_sum = combine(
        groupby(all_missing_trials, [experiment.participant_id_column, :session]),
        [:n_missing_trials, :n_trials] => ((x,t) -> sum(x) / sum(t)) => :prop_missing_all
    )
    
    # Unstack to wide format with task-specific missing proportions
    missing_trials = unstack(all_missing_trials, [experiment.participant_id_column, :session], :task, :prop_missing_trials; 
                           renamecols = x -> Symbol("prop_missing_$(x)"))
    leftjoin!(missing_trials, missing_trials_sum, on=[experiment.participant_id_column, :session])
    
    return (missing_trials, rts)
end

"""
    calculate_task_accuracies(jspsych_data)

Calculate accuracy metrics and critical values for reversal, PILT, and working memory tasks.
Returns tuple of (reversal_acc, pilt_acc, wm_acc, have_wm).
"""
function calculate_task_accuracies(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1,
    pilt_sequence::AbstractDataFrame,
    reversal_sequence::AbstractDataFrame
)
    # Get data for each task type
    reversal = filter(x -> x.trial_type == "reversal", jspsych_data)
    pilt = filter(x -> !ismissing(x.trialphase) && x.trialphase == "pilt", jspsych_data)
    
    # Reversal accuracy: only include valid responses
    reversal_acc = filter(x -> x.response in ["left", "right"], reversal)
    reversal_acc = combine(
        groupby(reversal_acc, [experiment.participant_id_column, :session]),
        :response_optimal => mean => :reversal_accuracy,
        [:response, :session] => ((r, s) -> reversal_critical_under_null(r; reversal_sequence = reversal_sequence, session = only(unique(s)))) => :reversal_critical_value
    )
    
    # PILT accuracy: only include valid responses and numeric blocks during main task
    pilt_acc = filter(x -> x.response in ["left", "right"] && isa(tryparse(Int64, string(x.block)), Number), pilt)
    pilt_acc = combine(
        groupby(pilt_acc, [experiment.participant_id_column, :session]),
        :response_optimal => (x -> round(mean(x), digits = 2)) => :pilt_accuracy,
        [:response, :session] => ((r, s) -> PILT_critical_under_null(r; pilt_sequence = pilt_sequence, session = only(unique(s)))) => :pilt_critical_value,
    )
    
    # Working memory accuracy (if available)
    have_wm = "wm" in filter(x -> !ismissing(x), unique(jspsych_data.trialphase))
    wm_acc = DataFrame()
    
    if have_wm
        wm = filter(x -> !ismissing(x.response) && (x.response in ["left", "middle", "right"]) && 
                      !ismissing(x.trialphase) && x.trialphase == "wm", jspsych_data)
        wm_acc = combine(
            groupby(wm, [experiment.participant_id_column, :session]),
            :response_optimal => (x -> round(mean(x), digits = 2)) => :wm_accuracy,
            [:optimal_side, :response] => ((o,r) -> WM_critical_under_null(o, r)) => :wm_critical_value,
        )
    end
    
    return (reversal_acc, pilt_acc, wm_acc, have_wm)
end

"""
    combine_accuracies_and_exclusion(reversal_acc, pilt_acc, wm_acc, have_wm)

Combine accuracy metrics and determine if participant meets performance criteria.
Creates 'any_accuracy' column indicating if participant performed above chance on any task.
"""
function combine_accuracies_and_exclusion(
    reversal_acc::DataFrame, 
    pilt_acc::DataFrame, 
    wm_acc::DataFrame, 
    have_wm::Bool;
    experiment::ExperimentInfo = TRIAL1
)
    if have_wm
        acc = outerjoin(reversal_acc, pilt_acc, wm_acc, on=[experiment.participant_id_column, :session])
        # Check if participant performed above critical value on any task
        acc.any_accuracy = (r -> ismissing(r.pilt_accuracy) || ismissing(r.reversal_accuracy) || ismissing(r.wm_accuracy) ||
                    ismissing(r.pilt_critical_value) || ismissing(r.reversal_critical_value) || ismissing(r.wm_critical_value) ? missing :
                    r.pilt_accuracy > r.pilt_critical_value || r.reversal_accuracy > r.reversal_critical_value || r.wm_accuracy > r.wm_critical_value).(eachrow(acc))
    else
        acc = outerjoin(reversal_acc, pilt_acc, on=[experiment.participant_id_column, :session])
    end
    
    return acc
end

"""
    calculate_additional_metrics(jspsych_data)

Calculate additional quality metrics including max press rate and browser interactions.
Returns tuple of (max_press_df, browser_interactions_df).
"""
function calculate_additional_metrics(
    jspsych_data::DataFrame;
    experiment::ExperimentInfo = TRIAL1
)
    # Max press rate from dedicated measurement task
    max_press = filter(x -> !ismissing(x.trialphase) && x.trialphase == "max_press_rate", jspsych_data)
    max_press = combine(
        groupby(max_press, [experiment.participant_id_column, :session]),
        :avgSpeed => maximum => :max_press_rate
    )
    
    # Browser interactions: focus loss and fullscreen exit events
    participant_sessions = unique(jspsych_data[:, [experiment.participant_id_column, :session]])
    
    # Parse browser interactions from various formats (JSON strings, arrays, etc.)
    browser_interactions = combine(
        groupby(jspsych_data, [experiment.participant_id_column, :session]),
        :browser_interactions => (x -> begin
            all_interactions = String[]
            for val in x
                if !ismissing(val)
                    if isa(val, String)
                        # Try parsing as JSON, fallback to string
                        try
                            parsed = JSON.parse(val)
                            if isa(parsed, Vector)
                                append!(all_interactions, string.(parsed))
                            else
                                push!(all_interactions, string(val))
                            end
                        catch
                            push!(all_interactions, string(val))
                        end
                    elseif isa(val, Vector)
                        append!(all_interactions, string.(val))
                    end
                end
            end
            all_interactions
        end) => :browser_interactions
    )
    
    # Count specific interaction types
    browser_interactions = combine(
        groupby(browser_interactions, [experiment.participant_id_column, :session]),
        :browser_interactions => (x -> sum(x .== "blur")) => :focus_loss_events,
        :browser_interactions => (x -> sum(x .== "fullscreenexit")) => :fullscreen_exit_events
    )
    
    # Ensure all participant/session combinations are included with 0 default
    browser_interactions = leftjoin(participant_sessions, browser_interactions, on=[experiment.participant_id_column, :session])
    browser_interactions.focus_loss_events = coalesce.(browser_interactions.focus_loss_events, 0)
    browser_interactions.fullscreen_exit_events = coalesce.(browser_interactions.fullscreen_exit_events, 0)
    
    return (max_press, browser_interactions)
end

"""
    quality_checks(; filter_func::Function = r -> true, records::AbstractVector)

Comprehensive data quality assessment for jsPsych experimental data.

Calculates multiple quality metrics including:
- Task completion times
- Missing trial proportions
- Reaction times (mean ± SD)
- Task accuracies and performance above chance
- Browser interaction events
- Exclusion criteria based on missing data, performance, and quiz attempts

# Arguments
- `filter_func`: Function to filter records (default: include all)
- `records`: Vector of records to process

# Returns
- DataFrame with quality metrics per participant/session, or `nothing` if no data
"""
function quality_checks(
    jspsych_data::DataFrame; 
    experiment::ExperimentInfo = TRIAL1,
    sequences_dir::String
)
    
    # Calculate completion times for each task
    completion_times = calculate_completion_times(jspsych_data; experiment = experiment)
    
    # Calculate PILT instruction quiz attempts
    pilt_quiz = calculate_pilt_quiz_attempts(jspsych_data; experiment = experiment)

    # Process individual tasks for missing trials and reaction times
    reversal_missing, reversal_rt = process_reversal_task(jspsych_data; experiment = experiment)
    pilt_missing, pilt_rt = process_pilt_task(jspsych_data; experiment = experiment)
    discounting_missing, discounting_rt = process_discounting_task(jspsych_data; experiment = experiment)
    control_missing = process_control_tasks(jspsych_data; experiment = experiment)
    vigour_pit_missing = process_vigour_and_pit_tasks(jspsych_data; experiment = experiment)

    # Combine all missing trials and reaction times data
    all_missing_trials = vcat(reversal_missing, pilt_missing, discounting_missing, control_missing, vigour_pit_missing)
    all_rts = vcat(reversal_rt, pilt_rt, discounting_rt)
    
    # Transform to wide format for analysis
    missing_trials, rts = consolidate_missing_trials_and_rts(all_missing_trials, all_rts; experiment = experiment)
    
    # Calculate task performance accuracies
    pilt_sequence, reversal_sequence = extract_all_sequences(joinpath(sequences_dir, experiment.project))

    reversal_acc, pilt_acc, wm_acc, have_wm = calculate_task_accuracies(jspsych_data; 
        experiment = experiment,
        pilt_sequence = pilt_sequence,
        reversal_sequence = reversal_sequence
    )
    acc = combine_accuracies_and_exclusion(reversal_acc, pilt_acc, wm_acc, have_wm; experiment = experiment)
    
    # Calculate additional behavioral metrics
    max_press, browser_interactions = calculate_additional_metrics(jspsych_data; experiment = experiment)
    
    # Create exclusion criteria based on multiple factors
    exclusion_criteria = outerjoin(
        missing_trials,
        acc,
        select(pilt_quiz, experiment.participant_id_column, :session, :n_pilt_quiz_attempts),
        on=[experiment.participant_id_column, :session]
    )
    
    # Add inclusion criteria for wk0 sessions (baseline assessment)
    if have_wm
        insertcols!(
            exclusion_criteria,
            3,
            :include => ifelse.(
                exclusion_criteria.session .!= "wk0",  # Only apply to baseline
                missing,
                # Include if: <10% missing, above-chance performance, ≤3 quiz attempts
                (r -> (ismissing(r.prop_missing_all) || ismissing(r.any_accuracy) || ismissing(r.n_pilt_quiz_attempts)) ? missing : 
                      (r.prop_missing_all < 0.1) && (r.any_accuracy == true) && (r.n_pilt_quiz_attempts <= 3)).(eachrow(exclusion_criteria))
            )
        )
    end
    
    # Combine all quality metrics into final output
    quality = outerjoin(
        exclusion_criteria,
        max_press,
        completion_times,
        rts,
        browser_interactions,
        on=[experiment.participant_id_column, :session],
        makeunique=true
    )
    
    return quality
end