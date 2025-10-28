# Setup
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase

function preprocess_control_data(
  control_task_df::DataFrame,
  control_report_df::DataFrame;
  experiment::ExperimentInfo=TRIAL1
)

  participant_id_column = experiment.participant_id_column
  # Create copies to avoid modifying originals
  task_df = copy(control_task_df)
  report_df = copy(control_report_df)

  # Handle empty report_df early
  if nrow(report_df) == 0
    # Still process task data to add prediction groups
    pred_trials = filter(row -> row.trialphase == "control_predict_homebase" && row.session != "screening", task_df)
    
    if !isempty(pred_trials)
      transform!(groupby(pred_trials, [participant_id_column, :session]),
        :n_control_trials => (x -> ceil.(Int, (x .- minimum(x) .+ 1) ./ 16)) => :prediction_group
      )
    
      task_with_groups = leftjoin(task_df,
        select(pred_trials, [participant_id_column, :session, :n_control_trials, :prediction_group]),
        on=[participant_id_column, :session, :n_control_trials])
      sort!(task_with_groups, [participant_id_column, :session, :n_control_trials])

    else
      task_with_groups = task_df
      sort!(task_with_groups, [participant_id_column, :session])
    end
    
    # Return empty DataFrames for confidence and controllability
    empty_confidence = DataFrame()
    empty_controllability = DataFrame()
    
    return task_with_groups, empty_confidence, empty_controllability
  end

  # Handle missing responses in control_report_data (convert nothing to missing)
  transform!(report_df, :response => (x -> ifelse.(.!(ismissing.(x)) .&& x .== nothing, missing, x)) => :response)

  # Create trial groups for prediction trials (every 4 consecutive trials form a group)
  # Filter for prediction trials first
  pred_trials = filter(row -> row.trialphase == "control_predict_homebase" && row.session != "screening", task_df)

  # Add prediction group variable (trials 7-10 = group 1, trials 24-27 = group 2, etc.; after every 17 trials for the column trial, which is the timeline variable)
  transform!(groupby(pred_trials, [participant_id_column, :session]),
    :trial => (x -> ceil.(Int, (x .- minimum(x) .+ 1) ./ 17)) => :prediction_group
  )

  # Merge prediction trials back with main task data
  task_with_groups = leftjoin(task_df,
    select(pred_trials, [participant_id_column, :session, :trial, :prediction_group]),
    on=[participant_id_column, :session, :trial])
  sort!(task_with_groups, [participant_id_column, :session, :trial])

  # Handle missing confidence ratings by matching with task trials
  confidence_df = filter(row -> row.trialphase == "control_confidence", report_df)

  # Ensure we capture trials that should have confidence ratings but are missing entirely from report_df

  # Left join to preserve all prediction trials, including those completely missing from report_df
  # Missing trials will have missing values for all report_df columns
  complete_confidence = leftjoin(
    select(pred_trials, [participant_id_column, :session, :trial, :prediction_group]), 
    confidence_df,
    on=[participant_id_column, :session, :trial])
  sort!(complete_confidence, [participant_id_column, :session, :trial])

  controllability_df = filter(row -> row.trialphase == "control_controllability", report_df)

  return task_with_groups, complete_confidence, controllability_df
end

function identify_missing_confidence_trials(
  complete_confidence_df::DataFrame;
  participant_id_column::Symbol=:participant_id
)
  """
  Helper function to identify which prediction trials are missing confidence ratings.
  Returns a DataFrame showing missing trials by participant and session.
  """
  missing_trials = filter(row -> ismissing(row.trialphase), complete_confidence_df)

  if nrow(missing_trials) > 0
    missing_summary = combine(
      groupby(missing_trials, [participant_id_column, :session]),
      :trial => (x -> sort(collect(x))) => :missing_trials,
      nrow => :n_missing
    )
    println("Found $(nrow(missing_trials)) missing confidence rating trials across $(nrow(missing_summary)) participant-session combinations")
    return missing_summary
  else
    println("No missing confidence rating trials found")
    return DataFrame()
  end
end

function plot_control_exploration_presses!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column
  # Filter for exploration trials
  explore_data = filter(row -> row.trialphase == "control_explore", df)

  # Remove missing trial_presses
  explore_data = dropmissing(explore_data, :trial_presses)

  # Calculate individual participant averages per current strength and factor
  group_cols = [participant_id_column, factor, :current]
  avg_group_cols = [factor, :current]

  individual_data = combine(
    groupby(explore_data, group_cols),
    :trial_presses => mean => :mean_trial_presses
  )
  sort!(individual_data, group_cols)

  # Group averages
  group_avg_data = combine(
    groupby(individual_data, avg_group_cols),
    :mean_trial_presses => mean => :avg_trial_presses,
    :mean_trial_presses => (x -> std(x) / sqrt(length(x))) => :se_trial_presses
  )

  # Add confidence bands
  transform!(group_avg_data,
    [:avg_trial_presses, :se_trial_presses] =>
      ByRow((avg, se) -> (avg - se, avg + se)) =>
        [:lower_bound, :upper_bound])
  sort!(group_avg_data, avg_group_cols)

  # Individual participant lines
  individual_mapping = mapping(
    :current => nonnumeric => "Current strength",
    :mean_trial_presses => "Trial presses",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:thin_alpha])

  # Group average with error bars
  group_plot = data(group_avg_data) * (
    mapping(
      :current => nonnumeric => "Current strength",
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
    mapping(
      :current => nonnumeric => "Current strength",
      :avg_trial_presses => "Trial presses",
      layout=factor
    ) * (visual(Scatter, color=:dodgerblue2, markersize=12) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
  )

  # Add reference lines for current strength thresholds
  threshold_df = DataFrame(y = [6, 12, 18], threshold = ["1: Low", "2: Mid", "3: High"])
  threshold_colors = [:gray75, :gray50, :gray25]  # Light to dark for Low to High
  threshold_plot = data(threshold_df) * mapping(:y, color = :threshold => scale(:secondary)) * visual(HLines, linestyle = :dot)

  # Combine plots
  final_plot = threshold_plot + individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)), secondary = (; palette = threshold_colors));
    axis=(xlabel="Current strength",
      ylabel="Trial presses"))

  Label(f[0, :], "Control: Trial Presses by Current Strength (Exploration)", tellwidth=false)

  return f
end

function plot_control_prediction_accuracy!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  prediction_group_column::Symbol=:prediction_group,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column
  # Filter for prediction trials and remove missing correct values
  pred_data = filter(row -> row.trialphase == "control_predict_homebase", df)
  dropmissing!(pred_data, :correct)

  # Split data into screening and non-screening sessions
  screening_data = filter(row -> row.session == "screening", pred_data)
  regular_data = filter(row -> row.session != "screening", pred_data)

  # Handle regular sessions (existing logic)
  if nrow(regular_data) > 0
    # Calculate accuracy by prediction group
    group_cols = [participant_id_column, factor, prediction_group_column]
    avg_group_cols = [factor, prediction_group_column]

    # Individual participant accuracy per prediction group
    individual_data = combine(
      groupby(regular_data, group_cols),
      :correct => mean => :accuracy
    )
    sort!(individual_data, group_cols)

    # Group averages
    group_avg_data = combine(
      groupby(individual_data, avg_group_cols),
      :accuracy => mean => :avg_accuracy,
      :accuracy => (x -> std(x) / sqrt(length(x))) => :se_accuracy
    )

    # Add confidence bands
    transform!(group_avg_data,
      [:avg_accuracy, :se_accuracy] =>
        ByRow((avg, se) -> (avg - se, avg + se)) =>
          [:lower_bound, :upper_bound])
    sort!(group_avg_data, avg_group_cols)

    # Individual participant lines
    individual_mapping = mapping(
      :prediction_group => "Prediction test group",
      :accuracy => "Prediction accuracy",
      color=participant_id_column,
      group=participant_id_column,
      layout=factor
    )

    individual_plot = data(individual_data) *
                      individual_mapping *
                      visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:thin_alpha])

    # Group average with error bars
    group_plot = data(group_avg_data) * (
      mapping(
        :prediction_group => "Prediction test group",
        :lower_bound, :upper_bound,
        layout=factor
      ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
      mapping(
        :prediction_group => "Prediction test group",
        :avg_accuracy => "Prediction accuracy",
        layout=factor
      ) * (visual(Scatter, color=:dodgerblue2, markersize=12) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
    )

    regular_plot = individual_plot + group_plot
  else
    regular_plot = data([]) * mapping() * visual(Lines)  # Empty plot
  end

  # Handle screening session (histogram of individual accuracies)
  if nrow(screening_data) > 0
    # Calculate overall accuracy per participant for screening (4 trials total)
    screening_individual = combine(
      groupby(screening_data, [participant_id_column]),
      :correct => mean => :accuracy
    )

    # Add session column for layout consistency
    screening_individual[!, :session] .= "screening"

    # Create histogram
    screening_plot = data(screening_individual) *
                     mapping(:accuracy, layout=:session) *
                     visual(Hist, bins=0:0.25:1, normalization=:none, color=:dodgerblue2)
  else
    screening_plot = data([]) * mapping() * visual(Hist)  # Empty plot
  end

  # Draw screening histogram (if data exists) - first row, shorter height
  if nrow(screening_data) > 0
    draw!(f[1, 1], screening_plot;
      axis=(xlabel="Screening prediction accuracy (4 trials)",
        ylabel="Count"))
  end

  # Draw regular sessions plot (if data exists) - second row
  if nrow(regular_data) > 0
    draw!(f[2, 1], regular_plot, scales(Color = (; palette = from_continuous(:roma)));
      axis=(xlabel="Prediction test group",
        ylabel="Prediction accuracy"))
    # Set row heights: screening (30%) and regular sessions (70%)
    rowsize!(f.layout, 1, Relative(0.3))
    rowsize!(f.layout, 2, Relative(0.7))
  end


  Label(f[0, :], "Control: Home Base Prediction Accuracy", tellwidth=false)

  return f
end

function plot_control_confidence_ratings!(
  f::Figure,
  confidence_df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  prediction_group_column::Symbol=:prediction_group,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column

  # Remove missing responses but keep track of them
  conf_data = dropmissing(confidence_df, :response)

  # Calculate individual participant averages per trial group and factor
  group_cols = [participant_id_column, factor, prediction_group_column]
  avg_group_cols = [factor, prediction_group_column]

  # Individual participant confidence per trial group
  individual_data = combine(
    groupby(conf_data, group_cols),
    :response => mean => :mean_confidence
  )
  sort!(individual_data, group_cols)

  # Group averages
  group_avg_data = combine(
    groupby(individual_data, avg_group_cols),
    :mean_confidence => mean => :avg_confidence,
    :mean_confidence => (x -> std(x) / sqrt(length(x))) => :se_confidence
  )

  # Add confidence bands
  transform!(group_avg_data,
    [:avg_confidence, :se_confidence] =>
      ByRow((avg, se) -> (avg - se, avg + se)) =>
        [:lower_bound, :upper_bound])
  sort!(group_avg_data, avg_group_cols)

  # Individual participant lines
  individual_mapping = mapping(
    :prediction_group => "Prediction test group",
    :mean_confidence => "Confidence rating",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:thin_alpha])

  # Group average with error bars
  group_plot = data(group_avg_data) * (
    mapping(
      :prediction_group => "Prediction test group",
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
    mapping(
      :prediction_group => "Prediction test group",
      :avg_confidence => "Confidence rating",
      layout=factor
    ) * (visual(Scatter, color=:dodgerblue2, markersize=12) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
  )

  # Combine plots
  final_plot = individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)));
    axis=(xlabel="Prediction test group",
      ylabel="Confidence rating (0-4)",
      yticks=0:4,
      limits=(nothing, (0, 4))))

  Label(f[0, :], "Control: Confidence Ratings After Prediction Trials", tellwidth=false)

  return f
end

function plot_control_controllability_ratings!(
  f::Figure,
  controllability_df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column

  # Remove missing responses but keep track of them
  ctrl_data = dropmissing(controllability_df, :response)

  # Calculate individual participant averages per trial and factor
  group_cols = [participant_id_column, factor, :trial]
  avg_group_cols = [factor, :trial]

  # Individual participant controllability per trial
  individual_data = combine(
    groupby(ctrl_data, group_cols),
    :response => mean => :mean_controllability
  )
  sort!(individual_data, group_cols)

  # Group averages
  group_avg_data = combine(
    groupby(individual_data, avg_group_cols),
    :mean_controllability => mean => :avg_controllability,
    :mean_controllability => (x -> std(x) / sqrt(length(x))) => :se_controllability
  )

  # Add confidence bands
  transform!(group_avg_data,
    [:avg_controllability, :se_controllability] =>
      ByRow((avg, se) -> (avg - se, avg + se)) =>
        [:lower_bound, :upper_bound])
  sort!(group_avg_data, avg_group_cols)

  # Individual participant lines
  individual_mapping = mapping(
    :trial => "Trial",
    :mean_controllability => "Controllability rating",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:thin_alpha])

  # Group average with error bars
  group_plot = data(group_avg_data) * (
    mapping(
      :trial => "Trial",
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
    mapping(
      :trial => "Trial",
      :avg_controllability => "Controllability rating",
      layout=factor
    ) * (visual(Scatter, color=:dodgerblue2, markersize=12) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
  )

  # Combine plots
  final_plot = individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)));
    axis=(
      xlabel="Trial",
      xticks=sort(unique(ctrl_data.trial)),
      ylabel="Controllability rating (0-4)",
      yticks=0:4,
      limits=(nothing, (0, 4))))

  Label(f[0, :], "Control: Controllability Ratings After Prediction Trials", tellwidth=false)

  return f
end

function plot_control_reward_rate_by_effort!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  x_variable::Symbol=:current,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column

  # Filter for reward trials and remove missing correct values
  reward_data = filter(row -> row.trialphase == "control_reward", df)
  dropmissing!(reward_data, :correct)
  dropmissing!(reward_data, x_variable)

  # Calculate individual participant reward rates per x_variable and factor
  group_cols = [participant_id_column, factor, x_variable]
  avg_group_cols = [factor, x_variable]

  # Individual participant reward rates per effort requirement
  individual_data = combine(
    groupby(reward_data, group_cols),
    :correct => mean => :reward_rate
  )
  sort!(individual_data, group_cols)

  # Group averages
  group_avg_data = combine(
    groupby(individual_data, avg_group_cols),
    :reward_rate => mean => :avg_reward_rate,
    :reward_rate => (x -> std(x) / sqrt(length(x))) => :se_reward_rate
  )

  # Add confidence bands
  transform!(group_avg_data,
    [:avg_reward_rate, :se_reward_rate] =>
      ByRow((avg, se) -> (avg - se, avg + se)) =>
        [:lower_bound, :upper_bound])
  sort!(group_avg_data, avg_group_cols)

  # Create axis labels based on x_variable
  x_label = if x_variable == :current
    "Current strength"
  elseif x_variable == :reward_amount
    "Reward amount"
  else
    string(x_variable)
  end

  # Individual participant lines
  individual_mapping = mapping(
    x_variable => nonnumeric => x_label,
    :reward_rate => "Reward rate",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:thin_alpha])

  # Group average with error bars
  group_plot = data(group_avg_data) * (
    mapping(
      x_variable => nonnumeric => x_label,
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
    mapping(
      x_variable => nonnumeric => x_label,
      :avg_reward_rate => "Reward rate",
      layout=factor
    ) * (visual(Scatter, color=:dodgerblue2, markersize=12) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
  )

  # Combine plots
  final_plot = individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)));
    axis=(xlabel=x_label,
      ylabel="Reward rate",
      limits=(nothing, (0, 1))))

  Label(f[0, :], "Control: Reward Rate by $x_label", tellwidth=false)

  return f
end

# Convenience functions that create new figures
function plot_control_exploration_presses(
  df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  figure_size=(800, 600)
)
  fig = Figure(size=figure_size)
  plot_control_exploration_presses!(fig, df; factor=factor, participant_id_column=participant_id_column)
  return fig
end

function plot_control_prediction_accuracy(
  df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  prediction_group_column::Symbol=:prediction_group,
  figure_size=(800, 600)
)
  # Create figure with custom row heights - screening shorter, regular sessions taller
  fig = Figure(size=figure_size)

  plot_control_prediction_accuracy!(fig, df; factor=factor, participant_id_column=participant_id_column, prediction_group_column=prediction_group_column)

  return fig
end

function plot_control_confidence_ratings(
  confidence_df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  prediction_group_column::Symbol=:prediction_group,
  figure_size=(800, 600)
)
  fig = Figure(size=figure_size)
  plot_control_confidence_ratings!(fig, confidence_df; factor=factor, participant_id_column=participant_id_column, prediction_group_column=prediction_group_column)
  return fig
end

function plot_control_controllability_ratings(
  controllability_df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  figure_size=(800, 600)
)
  fig = Figure(size=figure_size)
  plot_control_controllability_ratings!(fig, controllability_df; factor=factor, participant_id_column=participant_id_column)
  return fig
end

function plot_control_reward_rate_by_effort(
  df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  x_variable::Symbol=:current,
  figure_size=(800, 600)
)
  fig = Figure(size=figure_size)
  plot_control_reward_rate_by_effort!(fig, df; factor=factor, participant_id_column=participant_id_column, x_variable=x_variable)
  return fig
end