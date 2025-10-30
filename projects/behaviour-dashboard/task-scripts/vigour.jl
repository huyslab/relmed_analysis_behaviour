# Setup
using CairoMakie, AlgebraOfGraphics, DataFrames, StatsBase, StatsModels, MixedModels

function preprocess_vigour_data(
  df::DataFrame;
  participant_id_column::Symbol=:participant_id
)
  out_df = transform(df, [:trial_presses, :trial_duration] => ((x, y) -> x .* 1000 ./ y) => :press_per_sec)
  return out_df
end

function preprocess_vigour_test_data(
  df::DataFrame;
  participant_id_column::Symbol=:participant_id
)
  
  post_vigour_test_df = transform(
    df, 
    [:left_magnitude, :left_ratio, :right_magnitude, :right_ratio] => ((lm, lr, rm, rr) -> lm ./ lr .- rm ./ rr) => :Δrpp,
    :response => (x -> Int.(x .=== "ArrowLeft")) => :choice_left
  )
  return post_vigour_test_df
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

function plot_vigour_test_curve_by_rpp!(
  f::Figure,
  df::DataFrame;
  factor::Symbol=:session,
  experiment::ExperimentInfo=TRIAL1,
  config::Dict=plot_config
)

  participant_id_column = experiment.participant_id_column

  # Calculate accuracy metrics
  test_acc_df = select(df, [participant_id_column, factor, :Δrpp, :choice_left]) |>
                x -> transform(
    x,
    :choice_left => (x -> x .* 2 .- 1) => :choice_left,
    :Δrpp => ByRow(sign) => :correct_choice
  ) |>
                     x -> groupby(x, [participant_id_column, factor]) |>
                          x -> combine(
    x,
    [:choice_left, :correct_choice] => ((cl, cc) -> mean(cl .== cc)) => :accuracy
  )

  # Fit mixed-effects logistic regression model per session and generate predictions
  forml = FormulaTerm(
    Term(:choice_left),
    Term(:Δrpp) + (Term(:Δrpp) | Term(participant_id_column))
  )

  sessions = unique(df[!, factor])
  pred_effect_rpp_list = DataFrame[]

  for session in sessions
    df_session = filter(row -> row[factor] == session, df)
    isempty(df_session) && continue

    model = fit(
      GeneralizedLinearMixedModel,
      forml,
      df_session,
      Binomial()
    )

    Δrpp_range = LinRange(
      minimum(df_session.Δrpp),
      maximum(df_session.Δrpp),
      100
    )

    session_pred = DataFrame(
      participant_id_column => fill("New", length(Δrpp_range)),
      factor => fill(session, length(Δrpp_range)),
      :Δrpp => collect(Δrpp_range)
    )
    session_pred.choice_left = fill(0.5, nrow(session_pred))

    session_pred.pred = predict(
      model,
      session_pred;
      new_re_levels=:population,
      type=:response
    )

    push!(pred_effect_rpp_list, session_pred)
  end

  pred_effect_rpp = isempty(pred_effect_rpp_list) ?
                    DataFrame() :
                    vcat(pred_effect_rpp_list...; cols=:union)

  # Create plot
  p_choice_right = data(filter(row -> row.choice_left == 0, df)) *
                   mapping(:choice_left, :Δrpp, layout=factor) *
                   visual(RainClouds;
                     markersize=0,
                     color=RGBAf(7/255, 68/255, 31/255, 0.5),
                     plot_boxplots=false,
                     cloud_width=0.2,
                     clouds=hist,
                     orientation=:horizontal)
  p_choice_left = data(filter(row -> row.choice_left == 1, df)) *
                  mapping(:choice_left, :Δrpp, layout=factor) *
                  visual(RainClouds;
                    side=:right,
                    markersize=0,
                    color=RGBAf(63/255, 2/255, 73/255, 0.5),
                    plot_boxplots=false,
                    cloud_width=0.2,
                    clouds=hist,
                    orientation=:horizontal)
  p_curve = data(pred_effect_rpp) *
            mapping(:Δrpp, :pred, layout=factor) *
            visual(Lines, color=:gray50, linewidth=config[:thick_linewidth])

  p = p_choice_right + p_choice_left + p_curve

  draw!(f[1, 1],
    p;
    axis=(; xlabel="ΔRPP: Left option − Right option", ylabel="P(Choose left)"))
  Label(f[0, :], "Vigour Test: Choice Curve by ΔRPP", tellwidth=false)

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
