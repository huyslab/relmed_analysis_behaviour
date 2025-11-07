# Script to generate all figures and combine into markdown file
include("$(pwd())/core/experiment-registry.jl")

# Which experiment to generate the dashboard for
experiment_name = length(ARGS) > 0 ? ARGS[1] : "TRIAL1"
manual_download = length(ARGS) > 1 ? parse(Bool, ARGS[2]) : true
experiment = eval(Meta.parse(experiment_name))

# Setup
begin
    cd("/home/jovyan")

    using DataFrames, CairoMakie, Dates, CategoricalArrays

    # Include data scripts
    include("$(pwd())/core/preprocess_data.jl")

    script_dir = dirname(@__FILE__)

    # Load configurations and theme
    include(joinpath(script_dir, "config.jl"))

    # Markdown generation
    include(joinpath(script_dir, "utils/markdown.jl"))

    # Include task-specific scripts
    task_dir = joinpath(script_dir, "task-scripts")
    include(joinpath(task_dir, "card-choosing.jl"))
    include(joinpath(task_dir, "reversal.jl"))
    include(joinpath(task_dir, "delay-discounting.jl"))
    include(joinpath(task_dir, "vigour.jl"))
    include(joinpath(task_dir, "PIT.jl"))
    include(joinpath(task_dir, "control.jl"))
    include(joinpath(task_dir, "questionnaires.jl"))
    include(joinpath(task_dir, "pavlovian-lottery.jl"))

    # Include data quality scripts
    data_quality_dir = joinpath(script_dir, "data-quality")
    include(joinpath(data_quality_dir, "data-quality.jl"))

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results", experiment.project)
    isdir(result_dir) || mkpath(result_dir)
end

# Figure registry for markdown generation
figure_registry = Vector{NamedTuple{(:filename, :title), Tuple{String, String}}}()


# Register figure function
function register_save_figure(filename::String, f::Figure, title::String)
    push!(figure_registry, (filename = filename, title = title))
    save(joinpath(result_dir, filename * ".svg"), f)
    return f
end

# Load and preprocess data
begin 
    dat = preprocess_project(experiment; force_download = true, delay_ms = 65, use_manual_download = manual_download)
end

# Run quality checks
println("Running data quality checks...")
quality = quality_checks(
        dat.jspsych_data; 
        experiment = experiment,
        sequences_dir = joinpath(data_quality_dir, "task-sequences"),
        questionnaire = dat.questionnaire
    )

screening = filter(x -> x.session == "screening" && !ismissing(x.include) && x.include && Date(x.session_start_time) == Date(2025,10,28), quality)

wk0 = filter(x -> x.session == "wk0" && !ismissing(x.completion_time_quests) && x.PROLIFIC_PID in unique(screening.PROLIFIC_PID), quality)

for a in eachrow(wk0)
    println(a.PROLIFIC_PID, ",", round(a.total_bonus, digits=2))
end

# Generate PILT learning curve by session
let 

    if !haskey(dat, :PILT) || isempty(dat.PILT)
        return
    end

    PILT_main_sessions = filter(x -> x.session != "screening", dat.PILT)

    if isempty(PILT_main_sessions)
        return
    end

    println("Generating PILT learning curves...")

    # Plot by session
    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_facet!(f1, PILT_main_sessions; facet = :session, config = plot_config, experiment = experiment)

    filename = "PILT_learning_curves_by_session"
    register_save_figure(filename, f1, "PILT Learning Curves by Session")

    # Plot by session and valence
    PILT_main_sessions.valence = CategoricalArray(PILT_main_sessions.valence; ordered = true, levels = ["Reward", "Mixed", "Punishment"])
    f2 = Figure(size = (800, 600))
    plot_learning_curves_by_color_facet!(f2, PILT_main_sessions; facet = :session, color = :valence, color_label = "Valence", config = plot_config, experiment = experiment)
    filename2 = "PILT_learning_curves_by_session_and_valence"
    register_save_figure(filename2, f2, "PILT Learning Curves by Session and Valence")


    # Plot by block
    f3 = Figure(size = (1600, 800))
    plot_learning_curve_by_block!(
        f3,
        PILT_main_sessions;
        experiment = experiment
    )
    filename3 = "PILT_learning_curves_by_block"
    register_save_figure(filename3, f3, "PILT Learning Curves by Block")
end


# Generate WM learning curve by session
let 
    
    if !haskey(dat, :WM) || isempty(dat.WM)
        return
    end
    
    println("Generating Working Memory learning curves...")
    
    WM_main_sessions = filter(x -> x.session != "screening", dat.WM) |> x -> prepare_WM_data(x; experiment = experiment);

    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_facet!(
        f1,
        WM_main_sessions;
        facet = :session,
        xcol = :appearance,
        early_stopping_at = nothing,
        config = plot_config,
        experiment = experiment
        )

    filename1 = "WM_learning_curves_by_session"
    register_save_figure(filename1, f1, "Working Memory Learning Curves by Session")

    f2 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f2, WM_main_sessions; facet = :session, variability = :individuals, config = plot_config, experiment = experiment)
    filename2 = "WM_learning_curves_by_delay_bins_and_session_individuals"
    register_save_figure(filename2, f2, "Working Memory Learning Curves by Delay Bins and Session (Individual Participants)")

    f3 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f3, WM_main_sessions; facet = :session, config = plot_config, experiment = experiment)
    filename3 = "WM_learning_curves_by_delay_bins_and_session_group"
    register_save_figure(filename3, f3, "Working Memory Learning Curves by Delay Bins and Session (Group Average)")

end

# Generate reversal accuracy curve
let 

    if !haskey(dat, :reversal) || isempty(dat.reversal)
        return
    end
    
    println("Generating Reversal Learning accuracy curves...")

    preproc_df = preprocess_reversal_data(dat.reversal; experiment = experiment)

    f = Figure(size = (800, 600))

    plot_reversal_accuracy_curve_by_factor!(f, preproc_df; config = plot_config, experiment = experiment)

    filename = "reversal_accuracy_curve"
    register_save_figure(filename, f, "Reversal Learning Accuracy Curve")
end

# Generate delay discounting curves
let 

    if !haskey(dat, :delay_discounting) || isempty(dat.delay_discounting)
        return
    end

    println("Generating Delay Discounting curves...")

    preproc_df = preprocess_delay_discounting_data(dat.delay_discounting; experiment = experiment)

    sessions = sort(unique(preproc_df.session))
    dfs = [filter(x -> x.session == s, preproc_df) for s in sessions]
    model_names = ["delay_discounting_model_$(experiment.project)_$(s)" for s in sessions]

    fit_and_process(df; model_name, experiment) = post_process_dd_logistic_regression(
        fit_dd_logistic_regression(
            df;
            experiment = experiment,
            model_name = model_name
        ),
        experiment = experiment
    )
    fits = map((df, model_name) -> fit_and_process(df; model_name = model_name, experiment = experiment), dfs, model_names)

    # Add session column to each draw DataFrame and concatenate
    coef_draws = vcat([insertcols(fit, 1, :session => sessions[i]) for (i, fit) in enumerate(fits)]...)

    f = Figure(size = (800, 600))
    plot_value_ratio_as_function_of_delay!(f, coef_draws, preproc_df; config = plot_config, experiment = experiment)

    filename = "delay_discounting_curve_by_session"
    register_save_figure(filename, f, "Delay Discounting Curve by Session")
end

# Generate vigour plots
let 

    if !haskey(dat, :vigour) || isempty(dat.vigour)
        return
    end
    
    println("Generating Vigour plots...")
    
    vigour_processed = preprocess_vigour_data(dat.vigour)

    f1 = Figure(size = (800, 600))
    plot_vigour_press_rate_by_reward_rate!(f1, vigour_processed; factor=:session, config = plot_config, experiment = experiment)

    filename1 = "vigour_press_rate_by_reward_rate"
    register_save_figure(filename1, f1, "Vigour: Press Rate by Reward Rate")

    if !haskey(dat, :vigour_test) || isempty(dat.vigour_test)
        return
    end
    vigour_test_processed = preprocess_vigour_test_data(dat.vigour_test)
    f2 = Figure(size = (800, 600))
    plot_vigour_test_curve_by_rpp!(f2, vigour_test_processed; factor=:session, config = plot_config, experiment = experiment)

    filename2 = "vigour_test_curve_by_rpp"
    register_save_figure(filename2, f2, "Vigour: Test Curve by ΔRPP")
end

# Generate PIT plots
let 

    if !haskey(dat, :PIT) || isempty(dat.PIT)
        return
    end

    println("Generating PIT plots...")

    PIT_processed = preprocess_PIT_data(dat.PIT)

    f1 = Figure(size = (800, 600))
    plot_PIT_press_rate_by_coin!(f1, PIT_processed; factor=:session, config = plot_config, experiment = experiment)

    filename1 = "PIT_press_rate_by_pavlovian_stimuli"
    register_save_figure(filename1, f1, "PIT: Press Rate by Pavlovian Stimuli")

    # Check for PIT_test data before plotting
    if !haskey(dat, :PIT_test) || isempty(dat.PIT_test)
        println("Skipping PIT test accuracy plot: no data.")
    else
        f2 = Figure(size = (800, 600))
        plot_PIT_test_acc_by_valence!(f2, dat.PIT_test; factor=:session, config = plot_config, experiment = experiment)

        filename2 = "PIT_test_accuracy_by_valence"
        register_save_figure(filename2, f2, "PIT: Test Accuracy by Valence")
    end
end

# Generate control plots
let
    if !haskey(dat, :control) || isempty(dat.control)
        return
    end
    
    println("Generating Control task plots...")

    task_with_groups, complete_confidence, controllability_data = preprocess_control_data(
        dat.control.control_task, 
        dat.control.control_report; 
        experiment = experiment)

    # Exploration presses by current strength
    if !isempty(task_with_groups)
        f1 = Figure(size = (800, 600))
        plot_control_exploration_presses!(f1, task_with_groups; factor=:session, config = plot_config, experiment = experiment)
        filename1 = "control_exploration_presses_by_current_strength"
        register_save_figure(filename1, f1, "Control: Exploration Presses by Current Strength")

        # Prediction accuracy over time (with screening)
        f2 = Figure(size = (800, 800))
        plot_control_prediction_accuracy!(f2, task_with_groups; factor=:session, config = plot_config, experiment = experiment)
        filename2 = "control_prediction_accuracy_over_time"
        register_save_figure(filename2, f2, "Control: Prediction Accuracy Over Time")
    end

    # Confidence ratings
    if !isempty(complete_confidence)
        f3 = Figure(size = (800, 600))
        plot_control_confidence_ratings!(f3, complete_confidence; factor=:session, config = plot_config, experiment = experiment)
        filename3 = "control_confidence_ratings"
        register_save_figure(filename3, f3, "Control: Confidence Ratings Over Time")
    end

    # Controllability ratings
    if !isempty(controllability_data)
        f4 = Figure(size = (800, 600))
        plot_control_controllability_ratings!(f4, controllability_data; factor=:session, config = plot_config, experiment = experiment)
        filename4 = "control_controllability_ratings"
        register_save_figure(filename4, f4, "Control: Controllability Ratings Over Time")
    end

    if isempty(task_with_groups) || unique(task_with_groups.session) == ["screening"]
        return
    end

    # Reward rate by current strength (default)
    f5 = Figure(size = (800, 600))
    plot_control_reward_rate_by_effort!(f5, task_with_groups; factor=:session, x_variable=:current, config = plot_config, experiment = experiment)
    filename5 = "control_reward_rate_by_current_strength"
    register_save_figure(filename5, f5, "Control: Reward Rate by Current Strength")

    # Reward rate by reward amount
    f6 = Figure(size = (800, 600))
    plot_control_reward_rate_by_effort!(f6, task_with_groups; factor=:session, x_variable=:reward_amount, config = plot_config, experiment = experiment)
    filename6 = "control_reward_rate_by_reward_amount"
    register_save_figure(filename6, f6, "Control: Reward Rate by Reward Amount")
end

# Generate questionnaire histograms
let 

    if !haskey(dat, :questionnaire) || isempty(dat.questionnaire)
        return
    end

    println("Generating Questionnaire histograms...")

    f1 = Figure(size = (1200, 800))
    plot_questionnaire_histograms!(f1, dat.questionnaire; experiment = experiment)

    filename1 = "questionnaire_histograms"
    register_save_figure(filename1, f1, "Questionnaire Score Distributions")

    if !any(dat.questionnaire.trialphase .== "PVSS")
        return
    end

    f2 = Figure(size = (800, 600))
    plot_questionnaire_histograms!(f2, dat.questionnaire; columns = [:pvss_valuation, :pvss_expectancy, :pvss_effort, :pvss_anticipation, :pvss_responsiveness, :pvss_satiation], labels = ["Reward valuation", "Reward expectancy", "Effort valuation", "Reward anticipation", "Initial responsiveness", "Reward satiation"], experiment = experiment)

    filename2 = "pvss_domain_histograms"
    register_save_figure(filename2, f2, "PVSS Domain Distributions")
end

# Generate demographics barplots and histograms for norming samples
let 
    if !any(dat.questionnaire.trialphase .== "demographics")
        return
    end

    println("Generating Demographics overview...")

    f = Figure(size = (1600, 800))
    plot_demographics!(f, dat.questionnaire; experiment = experiment, factor=:sex)

    filename = "demographics"
    register_save_figure(filename, f, "Demographics Overview")
end

# Generate max press rate histogram
let 

    if !haskey(dat, :max_press) || isempty(dat.max_press)
        return
    end

    println("Generating Max Press Rate histogram...")

    max_press_clean = combine(groupby(dat.max_press, [experiment.participant_id_column, :session]), :avg_speed => maximum => :avg_speed)
    
    f = Figure(size = (800, 600))

    mp = data(max_press_clean) *
    mapping(
        :avg_speed => "Average Press Rate",
        layout = :session => "Session"
    ) * histogram(bins=20)

    plt = draw!(f[1, 1], mp; axis = (; ylabel = "# participants"))

    filename = "max_press_rate_histogram"
    register_save_figure(filename, f, "Max Press Rate Distribution by Session")
end

# Plot pavlovian lottery reaction times
let

    if !haskey(dat, :pavlovian_lottery) || isempty(dat.pavlovian_lottery)
        return
    end

    println("Generating Pavlovian Lottery reaction time plots...")

    f = Figure(size = (800, 600))
    plot_pavlovian_lottery_rt!(f, dat.pavlovian_lottery; config = plot_config, experiment = experiment)
    filename = "pavlovian_lottery_reaction_times"
    register_save_figure(filename, f, "Pavlovian Lottery Reaction Times by Pavlovian Value and Session")
end 

# Plot open text response lengths
let 

    if !haskey(dat, :open_text) || isempty(dat.open_text)
        return
    end

    println("Generating Open Text response length plots...")
    
    df = copy(dat.open_text)

    function count_words(str)
        if ismissing(str) || isempty(str) || isnothing(str)
            return 0
        end

        words = split(str, r"\W+")   # split on any non-word characters
        filter!(!isempty, words)     # remove empty strings
        return length(words)
    end

    df.response_length = count_words.(df.response)

    f = Figure(size = (800, 600))

    mp = data(df) *
    mapping(
        :response_length,
        row = :session,
        col = :question,
    ) * histogram(bins=20)

    plt = draw!(f[1, 1], mp; axis = (; xlabel = "Response Length (words)", ylabel = "# responses"))

    filename = "open_text_response_lengths"
    register_save_figure(filename, f, "Open Text Response Lengths by Session")
end

# Generate the dashboard
println("Generating markdown dashboard...")
generate_markdown_dashboard()


append_wide_table_to_readme(quality; result_dir = result_dir, title = "Data Quality Overview")

