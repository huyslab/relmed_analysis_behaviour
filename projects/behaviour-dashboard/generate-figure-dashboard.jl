# Script to generate all figures and combine into markdown file

# Setup
begin
    cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("$(pwd())/environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using DataFrames, CairoMakie, Dates, CategoricalArrays

    # Include data scripts
    include("$(pwd())/core/experiment-registry.jl")
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

    # Create output directory if it doesn't exist
    result_dir = joinpath(script_dir, "results")
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
    (; PILT, PILT_test, WM, WM_test, reversal, delay_discounting, vigour, PIT, max_press, control, questionnaire, pavlovian_lottery, open_text) = preprocess_project(TRIAL1; force_download = false)
end

# Generate PILT learning curve by session
let PILT_main_sessions = filter(x -> x.session != "screening", PILT)
    
    # Plot by session
    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_facet!(f1, PILT_main_sessions; facet = :session)

    filename = "PILT_learning_curves_by_session"
    register_save_figure(filename, f1, "PILT Learning Curves by Session")

    # Plot by session and valence
    PILT_main_sessions.valence = CategoricalArray(PILT_main_sessions.valence; ordered = true, levels = ["Reward", "Mixed", "Punishment"])
    f2 = Figure(size = (800, 600))
    plot_learning_curves_by_color_facet!(f2, PILT_main_sessions; facet = :session, color = :valence, color_label = "Valence")
    filename2 = "PILT_learning_curves_by_session_and_valence"
    register_save_figure(filename2, f2, "PILT Learning Curves by Session and Valence")
end

# Generate WM learning curve by session
let WM_main_sessions = filter(x -> x.session != "screening", WM) |> prepare_WM_data;

    f1 = Figure(size = (800, 600))
    plot_learning_curves_by_facet!(
        f1,
        WM_main_sessions;
        facet = :session,
        xcol = :appearance,
        early_stopping_at = nothing)

    filename1 = "WM_learning_curves_by_session"
    register_save_figure(filename1, f1, "Working Memory Learning Curves by Session")

    f2 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f2, WM_main_sessions; facet = :session, variability = :individuals)
    filename2 = "WM_learning_curves_by_delay_bins_and_session_individuals"
    register_save_figure(filename2, f2, "Working Memory Learning Curves by Delay Bins and Session (Individual Participants)")

    f3 = Figure(size = (800, 600))
    plot_learning_curve_by_delay_bins!(f3, WM_main_sessions; facet = :session)
    filename3 = "WM_learning_curves_by_delay_bins_and_session_group"
    register_save_figure(filename3, f3, "Working Memory Learning Curves by Delay Bins and Session (Group Average)")

end

# Generate reversal accuracy curve
let preproc_df = preprocess_reversal_data(reversal)

    f = Figure(size = (800, 600))

    plot_reversal_accuracy_curve_by_factor!(f, preproc_df)

    filename = "reversal_accuracy_curve"
    register_save_figure(filename, f, "Reversal Learning Accuracy Curve")
end

# Generate delay discounting curves
let preproc_df = preprocess_delay_discounting_data(delay_discounting)

    sessions = sort(unique(preproc_df.session))
    dfs = [filter(x -> x.session == s, preproc_df) for s in sessions]
    model_names = ["delay_discounting_model_$(s)" for s in sessions]

    fit_and_process = post_process_dd_logistic_regression ∘ fit_dd_logistic_regression
    fits = map((df, model_name) -> fit_and_process(df; model_name = model_name), dfs, model_names)

    # Add session column to each draw DataFrame and concatenate
    coef_draws = vcat([insertcols(fit, 1, :session => sessions[i]) for (i, fit) in enumerate(fits)]...)

    f = Figure(size = (800, 600))
    plot_value_ratio_as_function_of_delay!(f, coef_draws, preproc_df)

    filename = "delay_discounting_curve_by_session"
    register_save_figure(filename, f, "Delay Discounting Curve by Session")
end

# Generate vigour plots
let vigour_processed = preprocess_vigour_data(vigour)

    f1 = Figure(size = (800, 600))
    plot_vigour_press_rate_by_reward_rate!(f1, vigour_processed; factor=:session)

    filename1 = "vigour_press_rate_by_reward_rate"
    register_save_figure(filename1, f1, "Vigour: Press Rate by Reward Rate")
end

# Generate PIT plots
let PIT_processed = preprocess_PIT_data(PIT)

    f1 = Figure(size = (800, 600))
    plot_PIT_press_rate_by_coin!(f1, PIT_processed; factor=:session)

    filename1 = "PIT_press_rate_by_pavlovian_stimuli"
    register_save_figure(filename1, f1, "PIT: Press Rate by Pavlovian Stimuli")
end

# Generate control plots
let 
    task_with_groups, complete_confidence, controllability_data = preprocess_control_data(control.control_task, control.control_report)

    # Exploration presses by current strength
    f1 = Figure(size = (800, 600))
    plot_control_exploration_presses!(f1, task_with_groups; factor=:session)
    filename1 = "control_exploration_presses_by_current_strength"
    register_save_figure(filename1, f1, "Control: Exploration Presses by Current Strength")

    # Prediction accuracy over time (with screening)
    f2 = Figure(size = (800, 800))
    plot_control_prediction_accuracy!(f2, task_with_groups; factor=:session)
    filename2 = "control_prediction_accuracy_over_time"
    register_save_figure(filename2, f2, "Control: Prediction Accuracy Over Time")

    # Confidence ratings
    f3 = Figure(size = (800, 600))
    plot_control_confidence_ratings!(f3, complete_confidence; factor=:session)
    filename3 = "control_confidence_ratings"
    register_save_figure(filename3, f3, "Control: Confidence Ratings Over Time")

    # Controllability ratings
    f4 = Figure(size = (800, 600))
    plot_control_controllability_ratings!(f4, controllability_data; factor=:session)
    filename4 = "control_controllability_ratings"
    register_save_figure(filename4, f4, "Control: Controllability Ratings Over Time")

    # Reward rate by current strength (default)
    f5 = Figure(size = (800, 600))
    plot_control_reward_rate_by_effort!(f5, task_with_groups; factor=:session, x_variable=:current)
    filename5 = "control_reward_rate_by_current_strength"
    register_save_figure(filename5, f5, "Control: Reward Rate by Current Strength")

    # Reward rate by reward amount
    f6 = Figure(size = (800, 600))
    plot_control_reward_rate_by_effort!(f6, task_with_groups; factor=:session, x_variable=:reward_amount)
    filename6 = "control_reward_rate_by_reward_amount"
    register_save_figure(filename6, f6, "Control: Reward Rate by Reward Amount")
end

# Generate questionnaire histograms
let 
    f = Figure(size = (1200, 800))
    plot_questionnaire_histograms!(f, questionnaire; bins=15)

    filename = "questionnaire_histograms"
    register_save_figure(filename, f, "Questionnaire Score Distributions")
end

# Generate max press rate histogram
let max_press_clean = combine(groupby(max_press, [TRIAL1.participant_id_column, :session]), :avg_speed => maximum => :avg_speed)
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
    f = Figure(size = (800, 600))
    plot_pavlovian_lottery_rt!(f, pavlovian_lottery;)
    filename = "pavlovian_lottery_reaction_times"
    register_save_figure(filename, f, "Pavlovian Lottery Reaction Times by Pavlovian Value and Session")
end 

# Plot open text response lengths
let df = copy(open_text)

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
generate_markdown_dashboard()