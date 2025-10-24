# Setup
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase

function preprocess_vigour_data(
  df::DataFrame;
  participant_id_column::Symbol=:participant_id
)
  out_df = transform(df, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
  return out_df
end

function plot_vigour_press_rate_by_reward_rate!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  reward_column::Symbol=:reward_per_press,
  press_rate_column::Symbol=:press_per_sec,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column


  # Prepare the data - calculate individual participant averages per reward rate and factor
  group_cols = [participant_id_column, factor, reward_column]
  avg_group_cols = [factor, reward_column]

  # Individual participant averages
  individual_data = combine(
    groupby(df, group_cols),
    press_rate_column => mean => :mean_press_rate
  )
  sort!(individual_data, group_cols)

  # Group averages
  group_avg_data = combine(
    groupby(individual_data, avg_group_cols),
    :mean_press_rate => mean => :avg_press_rate,
    :mean_press_rate => (x -> std(x) / sqrt(length(x))) => :se_press_rate
  )
  # Add upper and lower bounds for the ribbon
  transform!(group_avg_data,
    [:avg_press_rate, :se_press_rate] =>
      ByRow((avg, se) -> (avg .- se, avg .+ se)) =>
        [:lower_bound, :upper_bound])
  sort!(group_avg_data, avg_group_cols)

  # Create individual participant lines mapping
  individual_mapping = mapping(
    reward_column => "Reward per press",
    :mean_press_rate => "Press rate (press/sec)",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  # Create individual participant lines (thin, semi-transparent)
  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:scatter_alpha])

  # Create group average lines (thick) with confidence bands
  group_plot = data(group_avg_data) * (
    mapping(
      reward_column => "Reward per press",
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Band, alpha=config[:band_alpha], color=:dodgerblue2) +
    mapping(
      reward_column => "Reward per press",
      :avg_press_rate => "Press rate (press/sec)",
      layout=factor
    ) * visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2)
  )

  # Combine plots
  final_plot = individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)));
    axis=(xlabel="Reward per press",
      ylabel="Press rate (press/sec)"))

  Label(f[0, :], "Vigour: Press Rate by Reward Rate", tellwidth=false)

  return f
end

function plot_vigour_press_rate_by_reward_rate(
  df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  reward_column::Symbol=:reward_per_press,
  press_rate_column::Symbol=:press_per_sec,
  figure_size=(800, 600) # in pixels
)

  # Create figure
  fig = Figure(size=figure_size)

  # Call the main plotting function
  plot_vigour_press_rate_by_reward_rate!(
    fig, df;
    factor=factor,
    participant_id_column=participant_id_column,
    reward_column=reward_column,
    press_rate_column=press_rate_column
  )

  return fig
end