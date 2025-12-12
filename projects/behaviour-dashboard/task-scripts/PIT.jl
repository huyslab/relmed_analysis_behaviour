# Setup
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase, CategoricalArrays, ColorSchemes

function preprocess_PIT_data(
  df::DataFrame;
  participant_id_column::Symbol=:participant_id
)
  # Create a copy to avoid modifying the original dataframe
  out_df = copy(df)

  # Calculate press per second
  transform!(out_df, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)

  # Create categorical variables for proper ordering
  transform!(out_df, :coin => (x -> categorical(x; levels=[-1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0], ordered=true)) => :coin_cat)
  transform!(out_df, :ratio => (x -> categorical(x; levels=[1, 8, 16], ordered=true)) => :ratio_cat)
  transform!(out_df, :magnitude => (x -> categorical(x; levels=[1, 2, 5], ordered=true)) => :magnitude_cat)

  return out_df
end

function plot_PIT_press_rate_by_coin!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  pavlovian_column::Symbol=:coin_cat,
  press_rate_column::Symbol=:press_per_sec,
  config::Dict = plot_config
)

  participant_id_column = experiment.participant_id_column

  # Prepare the data - calculate individual participant averages per pavlovian stimuli and factor
  group_cols = [participant_id_column, factor, pavlovian_column]
  avg_group_cols = [factor, pavlovian_column]

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
    pavlovian_column => "Pavlovian stimuli (coin value)",
    :mean_press_rate => "Press rate (press/sec)",
    color=participant_id_column,
    group=participant_id_column,
    layout=factor
  )

  # Create individual participant lines (thin, semi-transparent)
  individual_plot = data(individual_data) *
                    individual_mapping *
                    visual(Lines, linewidth=config[:thin_linewidth], alpha=config[:scatter_alpha])

  # Create group average lines (thick) with pointranges
  group_plot = data(group_avg_data) * (
    mapping(
      pavlovian_column => "Pavlovian stimuli (coin value)",
      :lower_bound, :upper_bound,
      layout=factor
    ) * visual(Rangebars, color=:dodgerblue2, linewidth=config[:thick_linewidth]) +
    mapping(
      pavlovian_column => "Pavlovian stimuli (coin value)",
      :avg_press_rate => "Press rate (press/sec)",
      layout=factor
    ) * (visual(Scatter, color=:dodgerblue2, markersize=config[:medium_markersize]) + visual(Lines, linewidth=config[:thick_linewidth], color=:dodgerblue2))
  )

  # Combine plots
  final_plot = individual_plot + group_plot

  # Draw the plot
  draw!(f[1, 1], final_plot, scales(Color = (; palette = from_continuous(:roma)));
    axis=(xlabel="Pavlovian stimuli (coin value)",
      ylabel="Press rate (press/sec)",
      xticklabelrotation=π/4))

  Label(f[0, :], "PIT: Press Rate by Pavlovian Stimuli", tellwidth=false)

  return f
end

function plot_PIT_test_acc_by_valence!(
    f::Figure,
    df::DataFrame;
    factor::Symbol=:session,
    experiment::ExperimentInfo=TRIAL1,
    config::Dict = plot_config
)
    participant_id_column = experiment.participant_id_column

    transform!(
      df,
      [:feedback_left, :feedback_right] => ByRow((ml, mr) ->
        ml * mr < 0 ? "Different" : (ml > 0 ? "Positive" : "Negative")) => :valence
    )
    dropmissing!(df, :response_optimal)
    acc_df = combine(
      groupby(df, [participant_id_column, factor, :valence]),
      :response_optimal => mean => :acc
    )

    # Create plot
    p = data(acc_df) *
      mapping(
        :acc => "Accuracy",
        color=:valence=>"Valence",
        layout=factor
      ) *
      histogram(Stairs; bins = 20)

    # Draw the plot
    plt = draw!(f[1, 1], p, scales(Color = (; palette=[colorant"gray50", ColorSchemes.Set1_5[[1,2]]...])))
    legend!(f[2, :], plt,
      titleposition = :left,
      tellwidth=false,
      halign=0.5,
      orientation=:horizontal,
      framevisible=false)

    Label(f[0, :], "PIT: Test Accuracy by Valence", tellwidth=false)

    return f
end

function plot_PIT_press_rate_by_coin(
  df::DataFrame;
  factor::Symbol=:session,
  participant_id_column::Symbol=:participant_id,
  pavlovian_column::Symbol=:coin_cat,
  press_rate_column::Symbol=:press_per_sec,
  figure_size=(800, 600) # in pixels
)

  # Create figure
  fig = Figure(size=figure_size)

  # Call the main plotting function
  plot_PIT_press_rate_by_coin!(
    fig, df;
    factor=factor,
    participant_id_column=participant_id_column,
    pavlovian_column=pavlovian_column,
    press_rate_column=press_rate_column
  )

  return fig
end